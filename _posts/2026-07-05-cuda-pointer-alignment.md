---
layout: post
title: 'How the “Correct” Way to Align CUDA Pointers Can Slow Down Shared-Memory Loads'
date: 2026-07-05 19:26:00
description: A pointer-to-integer round trip can erase CUDA shared-memory provenance, turn LDS/STS into generic LD.E/ST.E instructions, and slow down an otherwise correct kernel.
tags: CUDA MLsys
categories: Technology
---

## 1. The Problem Pattern: TL;DR

If your CUDA kernel partitions regions within dynamic shared memory and aligns them to 128, 512, or 1,024 bytes, a natural implementation is:

```text
shared pointer
-> convert to integer
-> round up with a bit mask
-> convert back to pointer
```

This pattern is common in CPU code, and the resulting address is numerically correct in CUDA as well.

The problem is that the CUDA compiler cares about more than the numerical value of an address. It must also determine which address space a pointer belongs to. If the final pointer is derived from an `extern __shared__` array through ordinary pointer arithmetic, the compiler can usually prove that it points to shared memory. An ordinary C++ dereference can then be compiled into a shared-memory instruction.

If the pointer is converted to an integer and later reconstructed from that integer, however, the compiler's dataflow chain is broken. We know that the address still lies in shared memory, but the compiler may not. With less information available, it uses more conservative—and less efficient—instructions, causing an unnecessary performance loss.

A more robust pattern is:

```text
shared pointer
-> use integer arithmetic only to compute misalignment / padding
-> return original pointer + padding
```

In other words, use the integer only to calculate how many more bytes are needed for alignment. The final result remains a pointer-arithmetic descendant of the original shared-memory pointer.

## 2. Minimal Reproduction

We can reproduce the issue with a deliberately simple kernel. The example contains two kernels that do the same work:

- Take a tile buffer from dynamic shared memory.
- Manually align the tile base.
- Repeatedly perform vectorized shared-memory loads in a hot loop, with a small amount of dependent computation.

The overall flow (the details are not important here) is:

```cpp
static constexpr int kSkew = 37;

extern "C" __global__ __launch_bounds__(kThreads, 1) void
roundtrip_bad_kernel(uint4 *out, uint32_t seed, int iters) { // or roundtrip_good_kernel
  extern __shared__ char smem[];
  char *base = bad_aligned_base(smem + kSkew); // or good_aligned_base
  uint4 *tile = reinterpret_cast<uint4 *>(base);

  init_shared(tile, seed);
  uint4 acc = load_loop(tile, iters, seed);
  out[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
```

Both kernels use the same load logic (also not important here):

```cpp
__device__ __forceinline__ uint4 load_loop(uint4 const *tile, int iters,
                                           uint32_t seed) {
  uint4 acc = make_uint4(seed + threadIdx.x, seed ^ threadIdx.x,
                         seed + 3u * threadIdx.x, seed ^ 0x5f3759dfu);

#pragma unroll 1
  for (int r = 0; r < iters; ++r) {
    int const idx = (threadIdx.x + r) & (kVecCount - 1);
    uint4 v = tile[idx];
    acc.x += v.x;
    acc.y ^= v.y;
    acc.z += v.z;
    acc.w ^= v.w;
  }

  return acc;
}
```

The only difference is how the pointer is aligned (this is the important part):

```cpp
static constexpr uintptr_t kAlign = 1024;

__device__ __forceinline__ char *bad_aligned_base(char *p) {
  uintptr_t raw = reinterpret_cast<uintptr_t>(p); // first convert to an integer
  raw = (raw + kAlign - 1) & ~(kAlign - 1);        // align
  return reinterpret_cast<char *>(raw);           // convert back to a pointer
}

__device__ __forceinline__ char *good_aligned_base(char *p) {
  uintptr_t const mis = reinterpret_cast<uintptr_t>(p) & (kAlign - 1);
  // Keep p itself as a shared-memory pointer and add only the padding offset.
  return p + ((kAlign - mis) & (kAlign - 1));
}
```

On a B200, the SASS summaries for the two implementations are:

```text
=== roundtrip_bad_kernel ===
LD.E.128: 1
LDS.128 : 0
ST.E.128: 5
STS.128 : 0

=== roundtrip_good_kernel ===
LD.E.128: 0
LDS.128 : 1
ST.E.128: 0
STS.128 : 5
```

The relevant SASS instructions are:

```text
// bad: generic load
/*0870*/ LD.E.128 R8, desc[UR8][R2.64] ;

// good: shared-memory load
/*0950*/ LDS.128 R8, [R0+UR4+0x25] ;
```

In the same benchmark run:

```text
bad  per_launch=314.396 us
good per_launch=270.530 us
ratio=1.162x
identical checksums
```

The numerical shared-memory address is the same, and the computation produces the same result. Yet the difference in pointer provenance alone changes the SASS from `LDS`/`STS` to `LD.E`/`ST.E`, and performance changes with it.

In the real performance-debugging scenario where I encountered this issue, the kernel runtime differed by as much as 1.5× between the two cases.

## 3. A Deeper Analysis

An ordinary pointer in CUDA C++ is a high-level-language pointer. At the PTX/SASS level, memory is divided into more explicit state spaces, including global, shared, local, constant, and generic.

The same piece of shared memory can have two address representations at different levels. One is the generic pointer used by ordinary CUDA C++ code. It is typically 64 bits wide, can be printed with `%p`, stored in a `uintptr_t`, and passed around as an ordinary pointer. The other is an address in PTX's `.shared` state space. It is more like an offset within the current cooperative thread array's (CTA's) shared-memory window and is typically a shorter raw offset. A generic address provides a unified representation that can map through different windows to global, shared, or local memory.

In our minimal kernel, for example, printing the start of dynamic shared memory with `printf("%p", smem)` in C++ produces a 64-bit generic pointer such as `0x7b4400000400`, while `__cvta_generic_to_shared(smem)` produces a raw shared offset such as `0x400`. Both refer to the same piece of shared memory, but they are not the same address representation. The high bits in `0x7b4400000400` can be understood as the encoding of the shared-memory window within the generic address space for that run, whereas `0x400` is the offset within the shared-memory window.

The bad and good paths can produce exactly the same runtime pointer value, and both can convert back to the same shared offset. The real difference is whether the compiler still knows that the address belongs to the `.shared` state space when lowering the code to PTX.

### The Paths Have Already Diverged in PTX

At the beginning of the PTX file, dynamic shared memory is declared as a `.shared` symbol. This CUDA declaration:

```cpp
extern __shared__ char smem[];
```

produces a PTX declaration like this:

```text
.extern .shared .align 16 .b8 smem[];
```

The key PTX pattern for the bad implementation is:

```text
mov.u32         %r63, smem;        // get the raw shared address of the PTX .shared symbol
cvt.u64.u32     %rd7, %r63;
cvta.shared.u64 %rd8, %rd7;        // shared raw address -> generic address
add.s64         %rd9, %rd8, 1060;  // add an offset to the generic 64-bit address
and.b64         %rd1, %rd9, -1024; // align the generic 64-bit address to 1024 bytes
...
add.s64         %rd21, %rd1, %rd20;
ld.v4.u32       {%r109, %r110, %r111, %r112}, [%rd21]; // generic PTX load
```

Notice that the final instruction is `ld.v4.u32`, with no `.shared` qualifier. The bad implementation is therefore already a generic load at the PTX level, and `ptxas` later lowers it to the SASS instruction `LD.E.128`.

The key PTX pattern for the good implementation is:

```text
mov.u32           %r121, smem;        // get the raw shared address of the PTX .shared symbol
...
and.b32           %r124, %r123, 1023;
add.s32           %r47, %r121, %r124; // compute padding on a shared 32-bit address expression
...
ld.shared.v4.u32  {%r128, %r129, %r130, %r131}, [%r127+37]; // shared PTX load
```

Here, the final instruction is `ld.shared.v4.u32`. The good implementation remains a shared-memory load at the PTX level, and `ptxas` later lowers it to the SASS instruction `LDS.128`. The runtime pointer values of the bad and good implementations can therefore be identical even though their intermediate code generation has already diverged.

### `LDS.128` vs. `LD.E.128`

`LDS.128` is a 128-bit load from the shared-memory window. It uses the dedicated shared-memory path, has a shorter address representation, and tells the scheduler explicitly that this is a shared-memory load.

`LD.E.128` is a generic load. In the generated SASS, it commonly uses a 64-bit address and a memory descriptor, for example:

```text
LD.E.128 ..., desc[...][...64]
```

Even when the generic address ultimately refers to shared memory, the compiler and scheduler must handle it more conservatively. In a hot loop dominated by shared-memory loads, this directly slows down the producer.

## 4. Conclusion

When accessing shared memory through ordinary C++ dereferences, try to keep the pointer as a direct pointer-arithmetic descendant of an `extern __shared__` or `__shared__` object.

Integer arithmetic is fine for calculating alignment padding, but do not reconstruct the final pointer by reinterpreting the integer. Return the original pointer plus the padding instead.

For hot loops, do not inspect only the CUDA source. Check the actual PTX and SASS as well. In particular, when correctness looks fine but performance does not, SASS is often more honest than the source code.

## References

- [NVIDIA CUDA C++ Programming Guide: Address Space Conversion Functions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [NVIDIA PTX ISA: `ld` and State-Space Addressing](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [NVIDIA PTX Writer's Guide to Interoperability: Pointer Sizes vs. Shared/Local Address Sizes](https://docs.nvidia.com/cuda/ptx-writers-guide-to-interoperability/index.html)
- [NVIDIA CUDA Binary Utilities: SASS Instruction Classes](https://docs.nvidia.com/cuda/archive/12.6.0/cuda-binary-utilities/index.html)
- [NVIDIA Developer Forums: Generic Pointer vs. Shared-Memory Pointer](https://forums.developer.nvidia.com/t/generic-pointer-vs-shared-memory-pointer/62546)
- [NVIDIA Developer Forums: Why `ldmatrix` Requires an Address in Shared Space](https://forums.developer.nvidia.com/t/why-do-i-need-to-convert-a-pointer-to-shared-address-space-before-using-the-ldmatrix-instruction/274466)

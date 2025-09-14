---
layout: post
date: 2025-06-18 15:59:00-0400
inline: true
related_posts: false
---

Our first version of Mirage Persistent Kernel is released:sparkles:! We developed a compiler that automatically transforms LLM inference into a single megakernel — a fused GPU kernel that performs all necessary computation and communication in one launch. This end-to-end GPU fusion approach reduces LLM inference latency by 1.2-6.7x. Have a look at the [blog](https://zhihaojia.medium.com/compiling-llms-into-a-megakernel-a-path-to-low-latency-inference-cf7840913c17) and the [codebase](https://github.com/mirage-project/mirage/tree/mpk).

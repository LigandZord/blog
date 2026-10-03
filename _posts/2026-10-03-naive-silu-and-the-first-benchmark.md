---
title: "Naive SiLU, and the first benchmark against PyTorch"
date: 2026-10-03 12:00:00 +0000
categories: [Deep Learning in Rust, Activation]
tags: [rust, simd, avx2, benchmarking]
math: true
---

<!--
Outline to fill in — replace each bracketed note with your own write-up.
-->

## Why this project

[A sentence or two on the Mamba motivation, and why this repo exists as the
on-ramp — naive -> layout -> SIMD -> threaded, learning the systems techniques
before applying them to the harder sequential-scan problem.]

## The naive implementation

[What SiLU is, the formula `x / (1 + exp(-x))`, why it's a good "hello SIMD"
warmup (pure elementwise, no neighbor dependence), and a short code snippet
of `silu_naive`.]

## Benchmarking methodology

[Flat `&[f32]` / `&mut [f32]` buffers instead of a shaped tensor type, since
activation doesn't care about logical shape — and why that stops being true
once you get to GEMM. Criterion on the Rust side, a matching torch CPU script
on the Python side, with tensor-clone/setup excluded from the timed region on
both sides so the comparison is apples-to-apples.]

## Results

[Naive Rust: 2.11 ms/iter (~0.50 G-elem/s) on 1,048,576 f32 elements.
Torch CPU, 1 thread: 821 us/iter (~1.28 G-elem/s).
Torch CPU, 11 threads (default): 333 us/iter (~3.14 G-elem/s).
~2.6x and ~6.3x slower respectively — and why that gap is the whole point:
torch's kernel is already AVX2-vectorized with a fast approximate `exp`,
naive Rust is a scalar loop calling libm's `exp` once per element.]

## What's next

[Heading into SIMD: AVX2 has no hardware `exp` instruction, so vectorizing
this isn't a 1:1 intrinsic swap — it means implementing a vectorized
approximation of `exp` itself, the Cephes/Pommier range-reduction +
polynomial technique. Tease the next post.]

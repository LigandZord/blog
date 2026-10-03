---
title: "Naive SiLU, and the first benchmark against PyTorch"
date: 2026-10-03 12:00:00 +0000
categories: [Deep Learning in Rust, Activation]
tags: [rust, simd, avx2, benchmarking]
math: true
---

## Why this project

The motivation for this project is to get an understanding of Rust, and why it is becoming popular, particularly in the deep learning community. And to do this, I plan on implementing different deep learning functions from scratch in Rust. The idea is, I get to learn the basics of Rust as a programming language, as well as try to understand other system-level concepts and CPU kernels that make the implementations faster. This is an exploratory series for me to enhance my knowledge of Rust and CPU kernels.

The end goal of the entire deep learning in Rust series is to implement the Mamba model in Rust from scratch, compare it with the torch implementation on CPU, and see how it stacks up. For each of the different deep learning modules, I will start with a basic implementation, then go on to progressively improve the speed with different system-level optimizations. To begin this series, I want to start with the implementation of an activation function, because I think that is probably the easiest starting point. I will then gradually implement other functions.

## The naive implementation

The activation function I plan on implementing first is the SiLU layer. This is because the authors of Mamba define it as their activation function. The definition itself is simple: `x / (1 + exp(-x))`.

For someone moving from Python to Rust, the input to the activation function itself was slightly confusing: should I take an array of a certain shape as input? How do I make it work for any input shape? When I explored this a little deeper, I realised that the shape itself is of no consequence: I can take the input as a slice, irrespective of what that shape is. Now why does this work? I saw that a tensor is normally stored in memory as one contiguous block. The shape is of consequence only to us while we do the math. Since the data is stored contiguously, that's also why reshape or view operations are so cheap on tensors.

Now, for the naive implementation, we just need to iterate through every element in the array and apply the SiLU formula above.

## Benchmarking methodology

For the naive kernel, I went with flat `&[f32]` / `&mut [f32]` slices rather than a shaped tensor type, which ties back to the point above. Since SiLU is a pure elementwise operation, it doesn't care about the logical shape of the tensor at all, only about the raw contiguous buffer underneath it. That stops being true once I get to GEMM, where the 2D structure of the matrix actually matters to the computation, but for activation it's a non-issue.

To benchmark, I used `criterion` on the Rust side and a matching PyTorch CPU script on the Python side, both run on the same input size: 1,048,576 `f32` elements. The output is written onto the same memory location in Rust. During the benchmarking, I made sure that the total time reported considers only the actual activation function calculation, and not other auxiliary tasks such as initializing the array, etc.

## Results

Here's where the naive version lands:

| Implementation | Time/iter | Throughput |
|---|---|---|
| `silu_naive` (Rust, scalar) | 2.11 ms | 0.50 G-elem/s |
| torch CPU, 1 thread | 821 µs | 1.28 G-elem/s |
| torch CPU, 11 threads (default) | 333 µs | 3.14 G-elem/s |

The naive Rust version is about 2.6x slower than PyTorch's single-threaded kernel, and about 6.3x slower than PyTorch's default multi-threaded run. That gap is expected, and honestly the whole point of this exercise: PyTorch's CPU kernel is already AVX2-vectorized with a fast approximate `exp`, while my naive version is just a scalar loop calling libm's `exp` once per element, with no vectorization and no threading. The single-threaded number is the fair target for where I am right now; the multi-threaded number is the stretch goal once `rayon` enters the picture later in this series.

## What's next

Next up is the SIMD version, and it's not as simple as swapping in a vector instruction. AVX2 has no hardware instruction for `exp` at all, so vectorizing this kernel means implementing a vectorized approximation of `exp` itself. More on that in the next post.

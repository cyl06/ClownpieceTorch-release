# ClownpieceTorch

**Teaching Assistants:** fAKe, MasterFHC

## Preface

This repository hosts the 2025 Summer Semester project for the Program Design and Data Structures course in SJTU's John Class.

The project aims to build a minimal PyTorch-style machine learning framework with C++ and Python.

As this project is newly developed, it may contain some imperfections or bugs. We encourage you to contact the TAs if you encounter any issues.

## Introduction

> I. What will you learn?

In this project, you will gain a deep understanding of the core mechanics behind modern deep learning frameworks like PyTorch. You will learn:
*   The fundamentals of `tensor` operations and how to implement a tensor library from scratch in C++.
*   The methodology to bind flexibilible Python frontend and performant C++ backend.
*   The concepts and princples behind the automatic differentiation engine.
*   The effective way to manage and construct neural network with module abstraction.
*   The basics of training a neural network, and its utilities like optimizer, scheduler, and dataloader.

Additionally, you will develop proficiency in Python programming.

> II. What will you build?

You will construct a functional mini deep learning framework similar to PyTorch:
*   **Week 1:** A multi-threaded CPU-based tensor library in C++ with Python bindings
*   **Week 2:** An autograd engine for differentiable operations implemented in Python
*   **Week 3:** A module system in Python to create and manage neural network components, and implementation of common modules(Linear, Conv2D, ReLU, etc.)
*   **Week 4:** A complete training pipeline with optimizers (SGD/Adam), schedulers, and dataloaders for building an image classification model

> III. What are the prerequisites?

Required:
*   Strong C++ programming skills (critical for Weeks 1 and 2)
*   Solid understanding of fundamental data structures
*   Basic calculus knowledge in the real domain, particularly derivatives and the chain rule

Helpful but not required:
*   Prior experience with machine learning or Python is beneficial; you'll build many ML concepts from the ground up, and Python is relatively easy to learn

> IV. How is the workload?

The project requires approximately 800 lines of code per week, which aligns with previous summer semester projects.

The first two weeks are in general more demanding, and minor delays are allowed.

Students with prior ML knowledge may find certain aspects less challenging, but will still gain valuable insights by implementing these fundamental components.

> V. What is the grading policy?

Your grade will primarily be determined by correctness of your code on the provided test sets (all available offline).

The first two weeks are in general more demanding, so minor delays are allowed.

The rest of grade depends on good coding style and final presentation.

> VI. Why "Clownpiece"?

<center>
<div style="width: 400px">
  <!-- <img src="docs/media/clownpiece.png" alt="Clownpiece"> -->
  <img src="https://en.touhouwiki.net/images/1/15/Hell_Whos_Who.png" alt="Clownpiece">
  
   <font color="gray"><p>Clownpiece appears at the bottom of the illustration.</p></font>
  <p><em>Credit: 東方Project人妖名鑑 宵暗篇 </br>Illustrator: 匡吉</em></p>

</div>
</center>

'Cause she holds a **torch**; that's it.

---

## Reference

Here are some references you may find helpful.

1. [Beginner Python Tutorial](https://docs.python.org/3/tutorial/index.html)
2. [Pytorch Tutorial](https://docs.pytorch.org/tutorials)
3. [Pytorch Documents](https://docs.pytorch.org/docs/stable/index.html)

More references will be provided in weekly tutorial documents.

---

## Future Work

We appreciate contributions from students with CUDA experience to include an optional CUDA tensor library challenge in next year's iteration of the project.

---

## License

Permission is granted to redistribute and adapt this work for pedagogical purposes without requiring prior consent from the authors.
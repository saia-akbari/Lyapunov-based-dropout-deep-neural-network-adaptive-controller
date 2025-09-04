[Read Me.txt](https://github.com/user-attachments/files/22141471/Read.Me.txt)
This repository contains the MATLAB code and functions for the ablation study presented in the paper: Lyapunov-Based Dropout Deep Neural Network (Lb-DDNN) Adaptive Controller.

The study investigates the performance of a novel Lb-DDNN adaptive controller designed to mitigate overfitting and co-adaptation in deep neural networks used for controlling nonlinear systems. The controller's weights are adapted online using a Lyapunov stability-driven law.  This repository provides the necessary scripts to reproduce the simulation results discussed in the paper.

The simulations for the ablation study are organized into different branches.  Each branch corresponds to a specific experiment detailed in the paper. To run a simulation, check out the relevant branch and run the main MATLAB script.

This repository uses different branches to organize the code for each part of the ablation study. Below is a guide to what each branch contains.

Lb-Dropout-DNN-vs.-Lb-DNN:
	Description: This is the main branch and contains the code for the 	primary comparison between the proposed Lb-DDNN controller (Lb-	DDNN 1) and the baseline Lb-DNN controller from (Lb-DNN 1).
Effect-of-Dropout-Frequency:
	Description: This branch is used to analyze the effect of varying 	the dropout frequency (deltat) on controller performance 	(simulations Lb-DDNN 2, 3, and 4).
Effect-of-Dropout-Deactivation-Time:
	Description: Investigates the impact of deactivating the dropout 	mechanism after the transient phase versus running it for the 	entire simulation (simulations Lb-DDNN 5 and 6).
Lb-Dropout-DNN-vs.-Sparse-DNN:
	Description: Compares the performance of two strategies for the 	steady-state phase: deactivating dropout completely (Lb-DDNN 7) 	versus preserving the last sparse network architecture (Lb-DDNN 	8).
Lb-Dropout-DNN-vs.-Narrow-DNN:
	Description: Compares the Lb-DDNN controller (Lb-DDNN 9) with a 	standard narrow Lb-DNN (Lb-DNN 2) that has a width equal to the 	number of active neurons in the dropout architecture.
Off-Trajectory-Function-approximation-Dropout-DNN-vs.-DNN:
	Description: Evaluates the generalization capability of the Lb-	DDNN (Lb-DDNN 10) compared to the baseline Lb-DNN (Lb-DNN 3) on an 	off-trajectory test dataset.

Effect-of-Number-of-Dropout-Neurons:
	Description: Examines how the number of active (non-dropped) 	neurons affects the overall system performance (Lb-DDNN 11).
Lb-Dropout-DNN-vs.-Lb-DNN-Second-Dynamics:
	Description: Validates the efficacy of the Lb-DDNN controller on a 	second, different nonlinear dynamical system (Lb-DDNN 12 vs. Lb-	DNN 4).
Lb-DDNN-vs.-pruning-DNN:
	Description: Compares the proposed Lb-DDNN controller (Lb-DDNN 13) 	with an online momentum-based pruning algorithm (Pruning DNN).

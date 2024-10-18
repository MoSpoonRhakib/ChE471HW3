# ChE471HW3
HW3: Improving Ising Model


**Responses for HW3**

1)
Tabulated breakdown of the code of line and memory profiler is shown in "Mohammed_Rhakib_HW3_Improving_Ising.ipynb"

The part that contributed the most to the execution time is the selection of random integers for random spin with coordinates (i, j)

The part the contributed to the most of the memory allocation is the spin flip once the criterion is met.



2)
Itemized list: Used np.roll() to avoid looping over each spin individually, lower precision integers to help save memory (this did not make a huge impact), and precomputed random variables to reduce computation time

Impact on execution time: For the L=10 case, the execution time decreased by ~.12 seconds based on the results from the line profiler.

Impact of memory: The memory allocation worsened (I do not understand why). With the optimizations, 149.6 MB was used for Q2 compared to 149.1 MB for Q1 (an increase of 0.5 MB)


3)
Code implementation of checkerboard and log-log plot shown in "Mohammed_Rhakib_HW3_Improving_Ising.ipynb". & "HW2.ipynb"



In terms of execution time, the checkerboard technique far exceeds the Ising model without the checkerbard. Using the line profilers, the average time it took to run the model from L = 1 to L = 10 was 0.71 seconds for checkerboard and 78.2 seconds for the Ising model without the checkerboard. This means that the checkerboard algorithm runs 110 faster than the old model. 

In terms of overhead, the overhead of the Ising model using the checkerboard algorithm is 2 magnitudes smaller than the Ising model without the checkerboard algorithm. (Overhead w/o checkerboard (9*10^-3 seconds) and overhead w/ checkerboard (1*10^-5 seconds)). The overhead is much smaller using the checkerboard algorithm.
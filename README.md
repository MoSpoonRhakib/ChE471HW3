# ChE471HW3
HW3: Improving Ising Model


**Responses for HW3**

1)
Tabulated breakdown of the code of line and memory profiler is shown in "Mohammed_Rhakib_HW3_Improving_Ising.ipynb"
For L=10, simulate_ising took 0.83 seconds and memory usages was 149.1 MB.

The part that contributed the most to the execution time is the selection of random integers for random spin with coordinates (i, j)

The part the contributed to the most of the memory allocation is the spin flip once the criterion is met.



2)
Itemized list: Used np.roll() to avoid looping over each spin individually, lower precision integers to help save memory (this did not make a huge impact), and precomputed random variables to reduce computation time

Impact on execution time: For the L=10 case, the execution time decreased by ~.12 seconds based on the results from the line profiler.

Impact of memory: The memory allocation worsened (I do not understand why). With the optimizations, 149.6 MB was used for Q2 compared to 149.1 MB for Q1 (an increase of 0.5 MB)


3)
Code implementation of checkerboard and log-log plot shown in "Mohammed_Rhakib_HW3_Improving_Ising.ipynb". & "HW2.ipynb"

In terms of execution time, the checkerboard technique far exceeds the Ising model without the checkerbard. Using the line profilers, the average time it took to run the model from L = 1 to L = 10 was 0.73 seconds for checkerboard and 80.4 seconds for the Ising model without the checkerboard. This means that the checkerboard algorithm runs 110 faster than the old model. 

In terms of overhead, the overhead of the Ising model using the checkerboard algorithm is 2 magnitudes smaller than the Ising model without the checkerboard algorithm. (Overhead w/o checkerboard (9x10^-3 seconds) and overhead w/ checkerboard (1x10^-5 seconds)). The overhead is much smaller using the checkerboard algorithm.

When comparing the time complexities of both modles, they both exhiubit O(N) notation. This means that both models exhibit linear time complexity. However, their slopes are differrent. The slope of the Ising model with the checkerboard algorithm is 100 times smaller than the old Ising model (3.77 x 10^-6 vs 4.70 x 10^-4). In conclusion, the checkerbaord algorthm was able to reduce the slope, the time it takes to complete the Ising model as the system size increases.
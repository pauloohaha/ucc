# Reduce-Scatter in TL/SHARP

Hello, in this branch we implement reduce-scatter by SHARP APIs and we would like to inquire wether our modifications are acceptable. 

We are three winner teams in the 11th RDMA Competition held by [HPC-AI Advisory Council](https://www.hpcadvisorycouncil.com/). We have further improved and organized the code developed by our three teams during the competition and conducted testing. Our SHARP-based implementation exhibits excellent communication performance compared to UCP's reduce-scatter.

## Implementation

We implement a `sharp_coll_do_allreduce_nb` based reduce-scatter and a `sharp_coll_do_reduce_nb` based one. Since the `sharp_coll_do_reduce_nb` API of the SHARP can only support messages with size >= 16K by default (this threshold can be configured). Thus, when the message size is small (total array size / number of nodes < 16K), the scatter-reduce is done based on the `sharp_coll_do_allreduce_nb` API of SHARP, when the message size is large (total array size / number of nodes >= 16K), the scatter-reduce is done based on the `sharp_coll_do_reduce_nb` API.


### Allreduce-Wrapped Reduce-Scatter

In the allreduce-wrapped reduce-scatter, we perform an allreduce operation on the send buffer for reduce-scatter, after which each process retrieves its data and places it into the receive buffer. If the IN_PLACE flag is not set, we need to allocate a temporary buffer to store the result of the allreduce operation.

### N-Reduce Reduce-Scatter

In the n-reduce reduce-scatter, we conduct n (the size of the UCC group) parallel reductions on the send buffer. During the reduction of the ith data, processer i serves as the root of this particular reduction.

## Evaluation

We conducted an evaluation of our reduce-scatter operation using the ucc_perftest benchmark on the Thor Cluster, which is provided by the HPC-AI Advisory Council. The hardware configuration of the Thor cluster is as follows:

- Dell™ PowerEdge™ R730 32-node cluster
- Dual Socket Intel® Xeon® 16-core CPUs E5-2697A V4 @ 2.60 GHz
- NVIDIA ConnectX-6 HDR100 100Gb/s InfiniBand adapter
- NVIDIA BlueField-2 SoC, HDR100 100Gb/s InfiniBand adapter
- NVIDIA BlueField-3 SoC, NDR200 200Gb/s InfiniBand adapter
- NVIDIA HDR Quantum Switch QM7800 40-Port 200Gb/s InfiniBand
- NVIDIA NDR Quantum-2 Switch QM9700 64-Port 400Gb/s InfiniBand
- Memory: 256GB DDR4 2400MHz RDIMMs per node
- 1TB 7.2K RPM SATA 2.5" hard drives per node

We utilized 16 nodes of the cluster and did not employ any BlueField DPU nodes. As demonstrated by the experimental results below, using Sharp significantly improves the performance of the reduce-scatter operation in all tested cases, compared to the current TL/UCP components.

In the following figures:

- 'no-sharp' represented by a black square indicates the use of UCP for reduce-scatter in TL.
- 'llt-arw' represented by a red circle indicates the use of SHARP-based allreduce-wrapped reduce-scatter in TL, specifically employing the low-latency tree.
- 'sat-arw' represented by a blue upper triangle indicates the use of SHARP-based allreduce-wrapped reduce-scatter in TL, specifically employing the streaming-aggregation tree.
- 'sat-nr' represented by a green lower triangle indicates the use of SHARP-based n-reduce reduce-scatter in TL, specifically employing the streaming-aggregation tree.

We observed that the `sharp_coll_do_reduce` function is worked only for streaming-aggregation tree when the message size is greater than or equal to 256B. Therefore, the n-reduce reduce-scatter is used on these message sizes.

4 nodes: 

![gnode4](https://redblog.oss-cn-chengdu.aliyuncs.com/picgo/gnode4.png)

8 nodes:

![gnode8](https://redblog.oss-cn-chengdu.aliyuncs.com/picgo/gnode8.png)

16 nodes:

![gnode16](https://redblog.oss-cn-chengdu.aliyuncs.com/picgo/gnode16.png)

Test python script:

```python
```



Raw data:

| count    | msg_size | avg      | min      | maximum  | nnode | ppn  | cl    | sharp    |
| -------- | -------- | -------- | -------- | -------- | ----- | ---- | ----- | -------- |
| 1        | 4        | 6.19     | 6.12     | 6.26     | 4     | 1    | basic | no-sharp |
| 2        | 8        | 7.44     | 7.37     | 7.48     | 4     | 1    | basic | no-sharp |
| 4        | 16       | 7.43     | 7.35     | 7.48     | 4     | 1    | basic | no-sharp |
| 8        | 32       | 7.46     | 7.43     | 7.5      | 4     | 1    | basic | no-sharp |
| 16       | 64       | 7.86     | 7.84     | 7.9      | 4     | 1    | basic | no-sharp |
| 32       | 128      | 8.3      | 8.26     | 8.35     | 4     | 1    | basic | no-sharp |
| 64       | 256      | 10.51    | 10.49    | 10.56    | 4     | 1    | basic | no-sharp |
| 128      | 512      | 11.11    | 11.06    | 11.2     | 4     | 1    | basic | no-sharp |
| 256      | 1024     | 11.35    | 11.3     | 11.39    | 4     | 1    | basic | no-sharp |
| 512      | 2048     | 12.33    | 12.3     | 12.36    | 4     | 1    | basic | no-sharp |
| 1024     | 4096     | 14.1     | 14.05    | 14.14    | 4     | 1    | basic | no-sharp |
| 2048     | 8192     | 18.95    | 18.9     | 19.02    | 4     | 1    | basic | no-sharp |
| 4096     | 16384    | 25.43    | 25.3     | 25.59    | 4     | 1    | basic | no-sharp |
| 8192     | 32768    | 38.85    | 38.74    | 39.09    | 4     | 1    | basic | no-sharp |
| 16384    | 65536    | 66.03    | 65.17    | 66.81    | 4     | 1    | basic | no-sharp |
| 32768    | 131072   | 114.58   | 114.23   | 115.2    | 4     | 1    | basic | no-sharp |
| 65536    | 262144   | 208.13   | 207.31   | 208.63   | 4     | 1    | basic | no-sharp |
| 131072   | 524288   | 404.05   | 401.44   | 406.25   | 4     | 1    | basic | no-sharp |
| 262144   | 1048576  | 776.91   | 771.95   | 780.08   | 4     | 1    | basic | no-sharp |
| 524288   | 2097152  | 1531.19  | 1511.4   | 1553.23  | 4     | 1    | basic | no-sharp |
| 1048576  | 4194304  | 3055.65  | 3037.47  | 3083.75  | 4     | 1    | basic | no-sharp |
| 2097152  | 8388608  | 8276.69  | 8244.52  | 8294.44  | 4     | 1    | basic | no-sharp |
| 4194304  | 16777216 | 26968.41 | 26677.17 | 27346.57 | 4     | 1    | basic | no-sharp |
| 8388608  | 33554432 | 53883.55 | 53193.68 | 54437.43 | 4     | 1    | basic | no-sharp |
| 16777216 | 67108864 | 107077.3 | 106020.6 | 108183.4 | 4     | 1    | basic | no-sharp |
| 1        | 4        | 2.7      | 2.68     | 2.73     | 4     | 1    | basic | llt-arw  |
| 2        | 8        | 2.74     | 2.73     | 2.75     | 4     | 1    | basic | llt-arw  |
| 4        | 16       | 2.76     | 2.75     | 2.78     | 4     | 1    | basic | llt-arw  |
| 8        | 32       | 2.87     | 2.86     | 2.88     | 4     | 1    | basic | llt-arw  |
| 16       | 64       | 3.42     | 3.38     | 3.48     | 4     | 1    | basic | llt-arw  |
| 32       | 128      | 3.86     | 3.84     | 3.9      | 4     | 1    | basic | llt-arw  |
| 64       | 256      | 4.18     | 4.16     | 4.21     | 4     | 1    | basic | llt-arw  |
| 128      | 512      | 5.02     | 5        | 5.04     | 4     | 1    | basic | llt-arw  |
| 256      | 1024     | 6.09     | 6.06     | 6.13     | 4     | 1    | basic | llt-arw  |
| 512      | 2048     | 8.03     | 7.97     | 8.06     | 4     | 1    | basic | llt-arw  |
| 1024     | 4096     | 15.55    | 15.53    | 15.56    | 4     | 1    | basic | llt-arw  |
| 2048     | 8192     | 29.19    | 29.17    | 29.2     | 4     | 1    | basic | llt-arw  |
| 4096     | 16384    | 56.37    | 56.33    | 56.39    | 4     | 1    | basic | llt-arw  |
| 8192     | 32768    | 109.17   | 108.96   | 109.39   | 4     | 1    | basic | llt-arw  |
| 16384    | 65536    | 214.74   | 213.79   | 215.12   | 4     | 1    | basic | llt-arw  |
| 32768    | 131072   | 425.95   | 424.76   | 426.51   | 4     | 1    | basic | llt-arw  |
| 65536    | 262144   | 850.26   | 850.13   | 850.43   | 4     | 1    | basic | llt-arw  |
| 131072   | 524288   | 1686.25  | 1686.13  | 1686.34  | 4     | 1    | basic | llt-arw  |
| 262144   | 1048576  | 3368.88  | 3368.66  | 3369.07  | 4     | 1    | basic | llt-arw  |
| 524288   | 2097152  | 6721.53  | 6721.09  | 6721.74  | 4     | 1    | basic | llt-arw  |
| 1048576  | 4194304  | 13473.55 | 13473    | 13474.7  | 4     | 1    | basic | llt-arw  |
| 2097152  | 8388608  | 29787.6  | 29785.33 | 29789.51 | 4     | 1    | basic | llt-arw  |
| 4194304  | 16777216 | 69329.36 | 69019.25 | 69635    | 4     | 1    | basic | llt-arw  |
| 8388608  | 33554432 | 144770.5 | 143827.5 | 145128   | 4     | 1    | basic | llt-arw  |
| 16777216 | 67108864 | 291667.9 | 291595.6 | 291788.5 | 4     | 1    | basic | llt-arw  |
| 1        | 4        | 2.69     | 2.67     | 2.71     | 4     | 1    | basic | sat-nr   |
| 2        | 8        | 2.74     | 2.72     | 2.76     | 4     | 1    | basic | sat-nr   |
| 4        | 16       | 2.74     | 2.73     | 2.75     | 4     | 1    | basic | sat-nr   |
| 8        | 32       | 2.86     | 2.82     | 2.88     | 4     | 1    | basic | sat-nr   |
| 16       | 64       | 5.17     | 5.16     | 5.22     | 4     | 1    | basic | sat-nr   |
| 32       | 128      | 5.23     | 5.2      | 5.24     | 4     | 1    | basic | sat-nr   |
| 64       | 256      | 5.88     | 5.14     | 6.55     | 4     | 1    | basic | sat-nr   |
| 128      | 512      | 6.02     | 5.28     | 6.7      | 4     | 1    | basic | sat-nr   |
| 256      | 1024     | 6.13     | 5.49     | 6.43     | 4     | 1    | basic | sat-nr   |
| 512      | 2048     | 6.81     | 5.97     | 7.18     | 4     | 1    | basic | sat-nr   |
| 1024     | 4096     | 7.18     | 6.05     | 7.79     | 4     | 1    | basic | sat-nr   |
| 2048     | 8192     | 8.17     | 7.18     | 9.14     | 4     | 1    | basic | sat-nr   |
| 4096     | 16384    | 10.46    | 9.81     | 11.69    | 4     | 1    | basic | sat-nr   |
| 8192     | 32768    | 15.88    | 15.12    | 17.21    | 4     | 1    | basic | sat-nr   |
| 16384    | 65536    | 27.53    | 26.84    | 29.18    | 4     | 1    | basic | sat-nr   |
| 32768    | 131072   | 51.06    | 50.34    | 52.43    | 4     | 1    | basic | sat-nr   |
| 65536    | 262144   | 97.85    | 97.13    | 99.52    | 4     | 1    | basic | sat-nr   |
| 131072   | 524288   | 220.13   | 219.58   | 221.6    | 4     | 1    | basic | sat-nr   |
| 262144   | 1048576  | 438.77   | 438.14   | 440.52   | 4     | 1    | basic | sat-nr   |
| 524288   | 2097152  | 882.82   | 882.07   | 884.84   | 4     | 1    | basic | sat-nr   |
| 1048576  | 4194304  | 1787.73  | 1786.91  | 1789.97  | 4     | 1    | basic | sat-nr   |
| 2097152  | 8388608  | 3652.78  | 3651.66  | 3655.79  | 4     | 1    | basic | sat-nr   |
| 4194304  | 16777216 | 7468.75  | 7467.69  | 7471.2   | 4     | 1    | basic | sat-nr   |
| 8388608  | 33554432 | 15137.65 | 15136.63 | 15140    | 4     | 1    | basic | sat-nr   |
| 16777216 | 67108864 | 30557.61 | 30556.8  | 30559.8  | 4     | 1    | basic | sat-nr   |
| 1        | 4        | 2.71     | 2.68     | 2.74     | 4     | 1    | basic | sat-arw  |
| 2        | 8        | 2.73     | 2.71     | 2.75     | 4     | 1    | basic | sat-arw  |
| 4        | 16       | 2.75     | 2.73     | 2.77     | 4     | 1    | basic | sat-arw  |
| 8        | 32       | 2.86     | 2.85     | 2.87     | 4     | 1    | basic | sat-arw  |
| 16       | 64       | 3.45     | 3.39     | 3.5      | 4     | 1    | basic | sat-arw  |
| 32       | 128      | 3.86     | 3.83     | 3.9      | 4     | 1    | basic | sat-arw  |
| 64       | 256      | 4.22     | 4.2      | 4.25     | 4     | 1    | basic | sat-arw  |
| 128      | 512      | 5        | 4.97     | 5.03     | 4     | 1    | basic | sat-arw  |
| 256      | 1024     | 6.09     | 6.08     | 6.11     | 4     | 1    | basic | sat-arw  |
| 512      | 2048     | 8.02     | 7.99     | 8.06     | 4     | 1    | basic | sat-arw  |
| 1024     | 4096     | 7.66     | 7.65     | 7.68     | 4     | 1    | basic | sat-arw  |
| 2048     | 8192     | 9.28     | 9.27     | 9.31     | 4     | 1    | basic | sat-arw  |
| 4096     | 16384    | 14.03    | 14.02    | 14.05    | 4     | 1    | basic | sat-arw  |
| 8192     | 32768    | 21.15    | 21.06    | 21.23    | 4     | 1    | basic | sat-arw  |
| 16384    | 65536    | 37.22    | 37.06    | 37.35    | 4     | 1    | basic | sat-arw  |
| 32768    | 131072   | 69.88    | 69.79    | 69.98    | 4     | 1    | basic | sat-arw  |
| 65536    | 262144   | 136.56   | 136.47   | 136.69   | 4     | 1    | basic | sat-arw  |
| 131072   | 524288   | 299.29   | 298.88   | 299.7    | 4     | 1    | basic | sat-arw  |
| 262144   | 1048576  | 571.52   | 570      | 573.22   | 4     | 1    | basic | sat-arw  |
| 524288   | 2097152  | 1155.31  | 1144.31  | 1166.91  | 4     | 1    | basic | sat-arw  |
| 1048576  | 4194304  | 2356.19  | 2332.09  | 2368.01  | 4     | 1    | basic | sat-arw  |
| 2097152  | 8388608  | 11591.81 | 11568.47 | 11607.25 | 4     | 1    | basic | sat-arw  |
| 4194304  | 16777216 | 25487.22 | 25450.08 | 25510.96 | 4     | 1    | basic | sat-arw  |
| 8388608  | 33554432 | 50639.96 | 50629.19 | 50655.21 | 4     | 1    | basic | sat-arw  |
| 16777216 | 67108864 | 101170.4 | 101140   | 101197   | 4     | 1    | basic | sat-arw  |
| 1        | 4        | 13.43    | 13.32    | 13.6     | 8     | 1    | basic | no-sharp |
| 2        | 8        | 17.3     | 17.15    | 17.48    | 8     | 1    | basic | no-sharp |
| 4        | 16       | 17.27    | 17.13    | 17.36    | 8     | 1    | basic | no-sharp |
| 8        | 32       | 17.2     | 17.08    | 17.32    | 8     | 1    | basic | no-sharp |
| 16       | 64       | 18.28    | 18.2     | 18.34    | 8     | 1    | basic | no-sharp |
| 32       | 128      | 19.13    | 19.02    | 19.24    | 8     | 1    | basic | no-sharp |
| 64       | 256      | 23.58    | 23.38    | 23.73    | 8     | 1    | basic | no-sharp |
| 128      | 512      | 24.71    | 24.57    | 24.83    | 8     | 1    | basic | no-sharp |
| 256      | 1024     | 25.45    | 25.38    | 25.54    | 8     | 1    | basic | no-sharp |
| 512      | 2048     | 27.6     | 27.47    | 27.77    | 8     | 1    | basic | no-sharp |
| 1024     | 4096     | 31.72    | 31.58    | 31.91    | 8     | 1    | basic | no-sharp |
| 2048     | 8192     | 42.33    | 42.1     | 42.55    | 8     | 1    | basic | no-sharp |
| 4096     | 16384    | 57.1     | 56.93    | 57.29    | 8     | 1    | basic | no-sharp |
| 8192     | 32768    | 88.38    | 87.95    | 88.97    | 8     | 1    | basic | no-sharp |
| 16384    | 65536    | 153.79   | 153.04   | 154.57   | 8     | 1    | basic | no-sharp |
| 32768    | 131072   | 265.07   | 264.42   | 265.71   | 8     | 1    | basic | no-sharp |
| 65536    | 262144   | 500.05   | 497.11   | 504.41   | 8     | 1    | basic | no-sharp |
| 131072   | 524288   | 921.76   | 917.62   | 924.64   | 8     | 1    | basic | no-sharp |
| 262144   | 1048576  | 1803.02  | 1794.68  | 1810.59  | 8     | 1    | basic | no-sharp |
| 524288   | 2097152  | 3532.39  | 3521.74  | 3545.42  | 8     | 1    | basic | no-sharp |
| 1048576  | 4194304  | 8078.49  | 8021.24  | 8150.99  | 8     | 1    | basic | no-sharp |
| 2097152  | 8388608  | 19526.98 | 19458.55 | 19597.24 | 8     | 1    | basic | no-sharp |
| 4194304  | 16777216 | 52539.24 | 52182.83 | 52916.2  | 8     | 1    | basic | no-sharp |
| 8388608  | 33554432 | 106060.2 | 105554.3 | 106625.8 | 8     | 1    | basic | no-sharp |
| 16777216 | 67108864 | 211120.6 | 209923.6 | 212278.3 | 8     | 1    | basic | no-sharp |
| 1        | 4        | 2.84     | 2.77     | 2.88     | 8     | 1    | basic | llt-arw  |
| 2        | 8        | 2.84     | 2.82     | 2.89     | 8     | 1    | basic | llt-arw  |
| 4        | 16       | 2.94     | 2.93     | 2.98     | 8     | 1    | basic | llt-arw  |
| 8        | 32       | 3.58     | 3.52     | 3.62     | 8     | 1    | basic | llt-arw  |
| 16       | 64       | 4.04     | 3.98     | 4.1      | 8     | 1    | basic | llt-arw  |
| 32       | 128      | 4.59     | 4.48     | 4.69     | 8     | 1    | basic | llt-arw  |
| 64       | 256      | 5.39     | 5.3      | 5.46     | 8     | 1    | basic | llt-arw  |
| 128      | 512      | 6.86     | 6.72     | 6.96     | 8     | 1    | basic | llt-arw  |
| 256      | 1024     | 9.26     | 9.1      | 9.37     | 8     | 1    | basic | llt-arw  |
| 512      | 2048     | 16.76    | 16.65    | 16.86    | 8     | 1    | basic | llt-arw  |
| 1024     | 4096     | 30.31    | 30.22    | 30.42    | 8     | 1    | basic | llt-arw  |
| 2048     | 8192     | 57.95    | 57.83    | 58.05    | 8     | 1    | basic | llt-arw  |
| 4096     | 16384    | 112.52   | 112.41   | 112.74   | 8     | 1    | basic | llt-arw  |
| 8192     | 32768    | 222.68   | 221.76   | 223.04   | 8     | 1    | basic | llt-arw  |
| 16384    | 65536    | 446.37   | 445.19   | 446.79   | 8     | 1    | basic | llt-arw  |
| 32768    | 131072   | 869.17   | 868.01   | 869.42   | 8     | 1    | basic | llt-arw  |
| 65536    | 262144   | 1737.31  | 1737.15  | 1737.46  | 8     | 1    | basic | llt-arw  |
| 131072   | 524288   | 3459.48  | 3459.25  | 3459.61  | 8     | 1    | basic | llt-arw  |
| 262144   | 1048576  | 6893.13  | 6892.53  | 6893.43  | 8     | 1    | basic | llt-arw  |
| 524288   | 2097152  | 13778.56 | 13777.59 | 13778.99 | 8     | 1    | basic | llt-arw  |
| 1048576  | 4194304  | 30467.59 | 30464.71 | 30468.94 | 8     | 1    | basic | llt-arw  |
| 2097152  | 8388608  | 69980.45 | 69815.62 | 70170.11 | 8     | 1    | basic | llt-arw  |
| 4194304  | 16777216 | 146219.4 | 145721.4 | 146409.1 | 8     | 1    | basic | llt-arw  |
| 8388608  | 33554432 | 297550.2 | 296459.5 | 297781.8 | 8     | 1    | basic | llt-arw  |
| 16777216 | 67108864 | 598079.1 | 597995.3 | 598157.5 | 8     | 1    | basic | llt-arw  |
| 1        | 4        | 2.82     | 2.8      | 2.84     | 8     | 1    | basic | sat-nr   |
| 2        | 8        | 2.83     | 2.81     | 2.87     | 8     | 1    | basic | sat-nr   |
| 4        | 16       | 2.93     | 2.9      | 2.96     | 8     | 1    | basic | sat-nr   |
| 8        | 32       | 5.3      | 5.25     | 5.35     | 8     | 1    | basic | sat-nr   |
| 16       | 64       | 5.47     | 5.42     | 5.52     | 8     | 1    | basic | sat-nr   |
| 32       | 128      | 5.64     | 5.54     | 5.68     | 8     | 1    | basic | sat-nr   |
| 64       | 256      | 6.94     | 5.98     | 7.91     | 8     | 1    | basic | sat-nr   |
| 128      | 512      | 7.01     | 6.01     | 7.99     | 8     | 1    | basic | sat-nr   |
| 256      | 1024     | 7.26     | 6.36     | 8.47     | 8     | 1    | basic | sat-nr   |
| 512      | 2048     | 7.88     | 6.66     | 8.85     | 8     | 1    | basic | sat-nr   |
| 1024     | 4096     | 8.78     | 7.63     | 10.11    | 8     | 1    | basic | sat-nr   |
| 2048     | 8192     | 10.93    | 10.01    | 12.76    | 8     | 1    | basic | sat-nr   |
| 4096     | 16384    | 16.5     | 15.83    | 18.62    | 8     | 1    | basic | sat-nr   |
| 8192     | 32768    | 27.54    | 26.99    | 29.39    | 8     | 1    | basic | sat-nr   |
| 16384    | 65536    | 50.75    | 50.16    | 52.9     | 8     | 1    | basic | sat-nr   |
| 32768    | 131072   | 97.8     | 97.25    | 99.69    | 8     | 1    | basic | sat-nr   |
| 65536    | 262144   | 220.3    | 219.77   | 221.9    | 8     | 1    | basic | sat-nr   |
| 131072   | 524288   | 436.89   | 436.31   | 438.75   | 8     | 1    | basic | sat-nr   |
| 262144   | 1048576  | 874.98   | 873.88   | 878.62   | 8     | 1    | basic | sat-nr   |
| 524288   | 2097152  | 1765.57  | 1765.04  | 1767.97  | 8     | 1    | basic | sat-nr   |
| 1048576  | 4194304  | 3613.11  | 3612.26  | 3615.26  | 8     | 1    | basic | sat-nr   |
| 2097152  | 8388608  | 7470     | 7469.32  | 7472.62  | 8     | 1    | basic | sat-nr   |
| 4194304  | 16777216 | 15264.72 | 15264.07 | 15267.5  | 8     | 1    | basic | sat-nr   |
| 8388608  | 33554432 | 30652.23 | 30651.58 | 30654.92 | 8     | 1    | basic | sat-nr   |
| 16777216 | 67108864 | 61581.56 | 61581    | 61584.29 | 8     | 1    | basic | sat-nr   |
| 1        | 4        | 2.81     | 2.79     | 2.83     | 8     | 1    | basic | sat-arw  |
| 2        | 8        | 2.78     | 2.76     | 2.8      | 8     | 1    | basic | sat-arw  |
| 4        | 16       | 2.93     | 2.91     | 2.97     | 8     | 1    | basic | sat-arw  |
| 8        | 32       | 3.6      | 3.54     | 3.63     | 8     | 1    | basic | sat-arw  |
| 16       | 64       | 4.02     | 3.99     | 4.07     | 8     | 1    | basic | sat-arw  |
| 32       | 128      | 4.6      | 4.48     | 4.7      | 8     | 1    | basic | sat-arw  |
| 64       | 256      | 5.37     | 5.26     | 5.49     | 8     | 1    | basic | sat-arw  |
| 128      | 512      | 6.8      | 6.64     | 6.93     | 8     | 1    | basic | sat-arw  |
| 256      | 1024     | 9.25     | 9.12     | 9.35     | 8     | 1    | basic | sat-arw  |
| 512      | 2048     | 7.53     | 7.47     | 7.57     | 8     | 1    | basic | sat-arw  |
| 1024     | 4096     | 8.85     | 8.8      | 8.93     | 8     | 1    | basic | sat-arw  |
| 2048     | 8192     | 13.07    | 13.03    | 13.12    | 8     | 1    | basic | sat-arw  |
| 4096     | 16384    | 19.25    | 19.18    | 19.35    | 8     | 1    | basic | sat-arw  |
| 8192     | 32768    | 33.2     | 33.07    | 33.26    | 8     | 1    | basic | sat-arw  |
| 16384    | 65536    | 61.51    | 61.37    | 61.81    | 8     | 1    | basic | sat-arw  |
| 32768    | 131072   | 118.47   | 118.25   | 118.67   | 8     | 1    | basic | sat-arw  |
| 65536    | 262144   | 264.26   | 263.82   | 264.96   | 8     | 1    | basic | sat-arw  |
| 131072   | 524288   | 529.61   | 528.04   | 531.62   | 8     | 1    | basic | sat-arw  |
| 262144   | 1048576  | 1042.82  | 1034.3   | 1050.54  | 8     | 1    | basic | sat-arw  |
| 524288   | 2097152  | 2132.34  | 2117.91  | 2141.33  | 8     | 1    | basic | sat-arw  |
| 1048576  | 4194304  | 11800.51 | 11776.94 | 11807.43 | 8     | 1    | basic | sat-arw  |
| 2097152  | 8388608  | 24546.88 | 24508.03 | 24561.12 | 8     | 1    | basic | sat-arw  |
| 4194304  | 16777216 | 49004.82 | 48967.96 | 49023.43 | 8     | 1    | basic | sat-arw  |
| 8388608  | 33554432 | 97557.92 | 97508.9  | 97593.09 | 8     | 1    | basic | sat-arw  |
| 16777216 | 67108864 | 197282.7 | 197184.7 | 197332.2 | 8     | 1    | basic | sat-arw  |
| 1        | 4        | 28.09    | 28       | 28.2     | 16    | 1    | basic | no-sharp |
| 2        | 8        | 37.06    | 36.9     | 37.32    | 16    | 1    | basic | no-sharp |
| 4        | 16       | 36.65    | 36.45    | 36.81    | 16    | 1    | basic | no-sharp |
| 8        | 32       | 36.57    | 36.41    | 36.74    | 16    | 1    | basic | no-sharp |
| 16       | 64       | 39.26    | 39       | 39.36    | 16    | 1    | basic | no-sharp |
| 32       | 128      | 40.71    | 40.57    | 40.85    | 16    | 1    | basic | no-sharp |
| 64       | 256      | 49.87    | 49.72    | 49.98    | 16    | 1    | basic | no-sharp |
| 128      | 512      | 52.58    | 52.32    | 53.03    | 16    | 1    | basic | no-sharp |
| 256      | 1024     | 53.91    | 53.73    | 54.04    | 16    | 1    | basic | no-sharp |
| 512      | 2048     | 58.54    | 58.38    | 58.71    | 16    | 1    | basic | no-sharp |
| 1024     | 4096     | 68.42    | 67.91    | 68.81    | 16    | 1    | basic | no-sharp |
| 2048     | 8192     | 90.78    | 89.42    | 91.59    | 16    | 1    | basic | no-sharp |
| 4096     | 16384    | 118.82   | 118.54   | 119.02   | 16    | 1    | basic | no-sharp |
| 8192     | 32768    | 186.22   | 185.7    | 186.6    | 16    | 1    | basic | no-sharp |
| 16384    | 65536    | 340.6    | 338.13   | 342.54   | 16    | 1    | basic | no-sharp |
| 32768    | 131072   | 565.27   | 564.46   | 566.37   | 16    | 1    | basic | no-sharp |
| 65536    | 262144   | 1028.68  | 1026.96  | 1031.7   | 16    | 1    | basic | no-sharp |
| 131072   | 524288   | 1985.03  | 1977     | 1993.01  | 16    | 1    | basic | no-sharp |
| 262144   | 1048576  | 3846.01  | 3834.95  | 3854.68  | 16    | 1    | basic | no-sharp |
| 524288   | 2097152  | 7816.11  | 7786.75  | 7864.7   | 16    | 1    | basic | no-sharp |
| 1048576  | 4194304  | 19832.7  | 19782.93 | 19873.64 | 16    | 1    | basic | no-sharp |
| 2097152  | 8388608  | 40917.34 | 40865.79 | 40989.82 | 16    | 1    | basic | no-sharp |
| 4194304  | 16777216 | 99975.54 | 99930.16 | 100036.9 | 16    | 1    | basic | no-sharp |
| 8388608  | 33554432 | 204123.8 | 203641.8 | 204790.1 | 16    | 1    | basic | no-sharp |
| 16777216 | 67108864 | 408446.7 | 407499.9 | 408887.6 | 16    | 1    | basic | no-sharp |
| 1        | 4        | 2.88     | 2.85     | 2.89     | 16    | 1    | basic | llt-arw  |
| 2        | 8        | 3.04     | 3        | 3.07     | 16    | 1    | basic | llt-arw  |
| 4        | 16       | 3.67     | 3.64     | 3.71     | 16    | 1    | basic | llt-arw  |
| 8        | 32       | 4.48     | 4.35     | 4.59     | 16    | 1    | basic | llt-arw  |
| 16       | 64       | 5.26     | 4.99     | 5.51     | 16    | 1    | basic | llt-arw  |
| 32       | 128      | 6.46     | 6.18     | 6.71     | 16    | 1    | basic | llt-arw  |
| 64       | 256      | 9.04     | 8.78     | 9.29     | 16    | 1    | basic | llt-arw  |
| 128      | 512      | 13.46    | 13.19    | 13.73    | 16    | 1    | basic | llt-arw  |
| 256      | 1024     | 21.92    | 21.64    | 22.19    | 16    | 1    | basic | llt-arw  |
| 512      | 2048     | 39.49    | 39.2     | 39.76    | 16    | 1    | basic | llt-arw  |
| 1024     | 4096     | 75.01    | 74.71    | 75.28    | 16    | 1    | basic | llt-arw  |
| 2048     | 8192     | 145.03   | 144.72   | 145.28   | 16    | 1    | basic | llt-arw  |
| 4096     | 16384    | 294.23   | 293.36   | 294.57   | 16    | 1    | basic | llt-arw  |
| 8192     | 32768    | 570.01   | 569.43   | 570.38   | 16    | 1    | basic | llt-arw  |
| 16384    | 65536    | 1126.69  | 1125.35  | 1127.16  | 16    | 1    | basic | llt-arw  |
| 32768    | 131072   | 2262.68  | 2260.92  | 2263.07  | 16    | 1    | basic | llt-arw  |
| 65536    | 262144   | 4510.45  | 4509.98  | 4510.95  | 16    | 1    | basic | llt-arw  |
| 131072   | 524288   | 8998.83  | 8998.5   | 8999.07  | 16    | 1    | basic | llt-arw  |
| 262144   | 1048576  | 17976.99 | 17975.99 | 17977.78 | 16    | 1    | basic | llt-arw  |
| 524288   | 2097152  | 38830.95 | 38827.66 | 38834.83 | 16    | 1    | basic | llt-arw  |
| 1048576  | 4194304  | 83524.61 | 83446.13 | 83628.15 | 16    | 1    | basic | llt-arw  |
| 2097152  | 8388608  | 172213.1 | 171953.2 | 172341.7 | 16    | 1    | basic | llt-arw  |
| 4194304  | 16777216 | 352505.2 | 351913.5 | 352617   | 16    | 1    | basic | llt-arw  |
| 8388608  | 33554432 | 705229.2 | 704044.3 | 705372.6 | 16    | 1    | basic | llt-arw  |
| 16777216 | 67108864 | 1419763  | 1419658  | 1419840  | 16    | 1    | basic | llt-arw  |
| 1        | 4        | 2.96     | 2.93     | 3        | 16    | 1    | basic | sat-nr   |
| 2        | 8        | 2.97     | 2.94     | 3.01     | 16    | 1    | basic | sat-nr   |
| 4        | 16       | 5.36     | 5.32     | 5.41     | 16    | 1    | basic | sat-nr   |
| 8        | 32       | 5.39     | 5.35     | 5.42     | 16    | 1    | basic | sat-nr   |
| 16       | 64       | 5.62     | 5.57     | 5.68     | 16    | 1    | basic | sat-nr   |
| 32       | 128      | 6.23     | 6.15     | 6.28     | 16    | 1    | basic | sat-nr   |
| 64       | 256      | 9.01     | 8.22     | 10.71    | 16    | 1    | basic | sat-nr   |
| 128      | 512      | 9.1      | 8.35     | 10.61    | 16    | 1    | basic | sat-nr   |
| 256      | 1024     | 9.53     | 8.81     | 11.15    | 16    | 1    | basic | sat-nr   |
| 512      | 2048     | 10.36    | 9.44     | 12.22    | 16    | 1    | basic | sat-nr   |
| 1024     | 4096     | 12.12    | 11.26    | 14       | 16    | 1    | basic | sat-nr   |
| 2048     | 8192     | 17.4     | 16.87    | 19.67    | 16    | 1    | basic | sat-nr   |
| 4096     | 16384    | 28.67    | 28.19    | 31.04    | 16    | 1    | basic | sat-nr   |
| 8192     | 32768    | 51.2     | 50.86    | 53.24    | 16    | 1    | basic | sat-nr   |
| 16384    | 65536    | 97.51    | 96.84    | 100.19   | 16    | 1    | basic | sat-nr   |
| 32768    | 131072   | 220.17   | 219.51   | 221.91   | 16    | 1    | basic | sat-nr   |
| 65536    | 262144   | 436.54   | 435.8    | 438.58   | 16    | 1    | basic | sat-nr   |
| 131072   | 524288   | 872.29   | 870.18   | 874.93   | 16    | 1    | basic | sat-nr   |
| 262144   | 1048576  | 1750.65  | 1749.74  | 1752.98  | 16    | 1    | basic | sat-nr   |
| 524288   | 2097152  | 3580.16  | 3579.45  | 3582.62  | 16    | 1    | basic | sat-nr   |
| 1048576  | 4194304  | 7389.65  | 7388.82  | 7392.82  | 16    | 1    | basic | sat-nr   |
| 2097152  | 8388608  | 15245.23 | 15244.39 | 15248.67 | 16    | 1    | basic | sat-nr   |
| 4194304  | 16777216 | 30918.23 | 30917.32 | 30921.83 | 16    | 1    | basic | sat-nr   |
| 8388608  | 33554432 | 61645.29 | 61644.59 | 61648.8  | 16    | 1    | basic | sat-nr   |
| 16777216 | 67108864 | 123365.5 | 123364.5 | 123368.7 | 16    | 1    | basic | sat-nr   |
| 1        | 4        | 2.91     | 2.89     | 2.94     | 16    | 1    | basic | sat-arw  |
| 2        | 8        | 2.93     | 2.9      | 2.95     | 16    | 1    | basic | sat-arw  |
| 4        | 16       | 3.61     | 3.56     | 3.66     | 16    | 1    | basic | sat-arw  |
| 8        | 32       | 4.4      | 4.27     | 4.5      | 16    | 1    | basic | sat-arw  |
| 16       | 64       | 5.27     | 4.99     | 5.49     | 16    | 1    | basic | sat-arw  |
| 32       | 128      | 6.36     | 6.12     | 6.58     | 16    | 1    | basic | sat-arw  |
| 64       | 256      | 8.88     | 8.6      | 9.16     | 16    | 1    | basic | sat-arw  |
| 128      | 512      | 13.11    | 12.85    | 13.35    | 16    | 1    | basic | sat-arw  |
| 256      | 1024     | 7.4      | 7.34     | 7.47     | 16    | 1    | basic | sat-arw  |
| 512      | 2048     | 8.67     | 8.58     | 8.77     | 16    | 1    | basic | sat-arw  |
| 1024     | 4096     | 12.57    | 12.52    | 12.62    | 16    | 1    | basic | sat-arw  |
| 2048     | 8192     | 18.32    | 18.2     | 18.4     | 16    | 1    | basic | sat-arw  |
| 4096     | 16384    | 31.27    | 31.22    | 31.36    | 16    | 1    | basic | sat-arw  |
| 8192     | 32768    | 57.26    | 57.17    | 57.36    | 16    | 1    | basic | sat-arw  |
| 16384    | 65536    | 109.79   | 109.65   | 109.94   | 16    | 1    | basic | sat-arw  |
| 32768    | 131072   | 246.16   | 245.76   | 246.53   | 16    | 1    | basic | sat-arw  |
| 65536    | 262144   | 494.95   | 494      | 495.77   | 16    | 1    | basic | sat-arw  |
| 131072   | 524288   | 997.77   | 994.1    | 1002.34  | 16    | 1    | basic | sat-arw  |
| 262144   | 1048576  | 2013.23  | 2001.89  | 2019.63  | 16    | 1    | basic | sat-arw  |
| 524288   | 2097152  | 11931.21 | 11913.41 | 11937.93 | 16    | 1    | basic | sat-arw  |
| 1048576  | 4194304  | 24107.37 | 24075.58 | 24123.03 | 16    | 1    | basic | sat-arw  |
| 2097152  | 8388608  | 48147.2  | 48092.65 | 48171.82 | 16    | 1    | basic | sat-arw  |
| 4194304  | 16777216 | 95934.55 | 95887.15 | 95954.49 | 16    | 1    | basic | sat-arw  |
| 8388608  | 33554432 | 193929.4 | 193893   | 193972.7 | 16    | 1    | basic | sat-arw  |
| 16777216 | 67108864 | 408847.9 | 408792.2 | 408926   | 16    | 1    | basic | sat-arw  |

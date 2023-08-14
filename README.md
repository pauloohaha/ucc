## CUHK Team

## Run Our Code

### UCC Compile

```bash

module load gcc hpcx
rm -rf UCC; mkdir UCC; cd UCC
git clone -b rc_reducenb_implemented https://github.com/pauloohaha/ucc.git
cd ucc
# compile
./autogen.sh; ./configure --prefix=/global/home/users/rdmaworkshop01/CUHK/piao/UCC/ --with-ucx=/global/software/rocky-9.x86_64/modules/gcc/11/hpcx/2.15/ucx --with-mpi=/global/software/rocky-9.x86_64/modules/gcc/11/hpcx/2.15/ompi

CFLAGS="-g -O0"
make -j install
```

### Env Setup

```bash
salloc -N 4 --partition thor --time=1:00:00 --nodelist thor0[25,32]

export LIBRARY_PATH=/global/home/users/rdmaworkshop01/CUHK/piao/UCC/lib:$LIBRARY_PATH

export HPCX_UCC_DIR=/global/home/users/rdmaworkshop01/CUHK/piao/UCC/

export LD_LIBRARY_PATH=/global/home/users/rdmaworkshop01/CUHK/piao/UCC/lib:$LD_LIBRARY_PATH

export CPATH=/global/home/users/rdmaworkshop01/CUHK/piao/UCC/include:$CPATH
```

### Commands

```bash
# with sharp
mpirun  -x UCC_CL_BASIC_TLS=ucp,sharp -x UCC_CLS=basic -x UCC_TL_SHARP_LOG_LEVEL=INFO -x UCC_LOG_LEVEL=ERROR -map-by ppr:1:node -report-bindings -mca pml ucx -mca coll_hcoll_enable 0 -x LD_LIBRARY_PATH -x ENABLE_SHARP_COLL=1 -x SHARP_COLL_ENABLE_SAT=1 -x SHARP_COLL_LOG_LEVEL=3 ucc_perftest -c reduce_scatter -F -b $((16*1024)) -e $((1280*1024*1024)) -w 100
# without sharp (ucp)
mpirun -x UCC_CL_BASIC_TLS=ucp -x UCC_CLS=basic -x UCC_TL_SHARP_LOG_LEVEL=ERROR -x UCC_LOG_LEVEL=ERROR -map-by ppr:1:node -report-bindings -mca pml ucx -mca coll_hcoll_enable 0 -x LD_LIBRARY_PATH -x ENABLE_SHARP_COLL=1 -x SHARP_COLL_ENABLE_SAT=1 -x SHARP_COLL_LOG_LEVEL=3 ucc_perftest -c reduce_scatter -F -b $((16*1024)) -e $((128*1024*1024)) -w 100
```

## Brief Report

### Code Modification to UCC

We added `init` and `start` function to `ucc/src/components/tl/sharp/tl_sharp_coll.c` as follow, it implemented `reduce_scatter` using ucc/sharp with reduce operation. We also modified various files to register this new operation to ucc library.

```c 
ucc_status_t ucc_tl_sharp_reduce_scatter_start(ucc_coll_task_t *coll_task)
{   
    ucc_info( "*********** sharp_reduce_scatter_start ************\n");
    ucc_tl_sharp_task_t          *task  = ucc_derived_of(coll_task, ucc_tl_sharp_task_t);
    ucc_tl_sharp_team_t          *team  = TASK_TEAM(task);
    ucc_coll_args_t              *args  = &TASK_ARGS(task);
    size_t                        count = args->dst.info.count;
    ucc_datatype_t                dt    = args->dst.info.datatype;
    struct sharp_coll_reduce_spec reduce_spec;
    enum sharp_datatype           sharp_type;
    enum sharp_reduce_op          op_type;
    size_t                        data_size;
    int                           ret;

    int              rank = (int)(coll_task->bargs.team->rank);
    int              size = (int)(coll_task->bargs.team->size);

    //initialize sharp_req hands
    void **sharp_reqs;
    sharp_reqs = malloc(sizeof(void *)*size);


    UCC_TL_SHARP_PROFILE_REQUEST_EVENT(coll_task, "sharp_reduce_scatter_start", 0); // Not sure

    sharp_type = ucc_to_sharp_dtype[UCC_DT_PREDEFINED_ID(dt)];
    op_type    = ucc_to_sharp_reduce_op[args->op];
    data_size  = ucc_dt_size(dt) * count;

    /*offset for each scatter*/
    long long offset = ((long long) data_size)/size;

    if (!UCC_IS_INPLACE(*args)) {
        ucc_tl_sharp_mem_register(TASK_CTX(task), team, args->src.info.buffer,data_size,
                                  &task->allreduce.s_mem_h);
    }
    ucc_tl_sharp_mem_register(TASK_CTX(task), team, args->dst.info.buffer, data_size,
                              &task->allreduce.r_mem_h);

    if (!UCC_IS_INPLACE(*args)) {
        reduce_spec.sbuf_desc.buffer.ptr        = args->src.info.buffer;
        reduce_spec.sbuf_desc.buffer.mem_handle = task->allreduce.s_mem_h->mr;
        reduce_spec.sbuf_desc.mem_type          = ucc_to_sharp_memtype[args->src.info.mem_type];
    } else {
        reduce_spec.sbuf_desc.buffer.ptr        = args->dst.info.buffer;
        reduce_spec.sbuf_desc.buffer.mem_handle = task->allreduce.r_mem_h->mr;
        reduce_spec.sbuf_desc.mem_type          = ucc_to_sharp_memtype[args->dst.info.mem_type];
    }

    reduce_spec.sbuf_desc.buffer.length     = data_size;
    reduce_spec.sbuf_desc.type              = SHARP_DATA_BUFFER;
    reduce_spec.rbuf_desc.buffer.ptr        = args->dst.info.buffer;
    reduce_spec.rbuf_desc.buffer.length     = data_size;
    reduce_spec.rbuf_desc.buffer.mem_handle = task->allreduce.r_mem_h->mr;
    reduce_spec.rbuf_desc.type              = SHARP_DATA_BUFFER;
    reduce_spec.rbuf_desc.mem_type          = ucc_to_sharp_memtype[args->dst.info.mem_type];
    reduce_spec.aggr_mode                   = SHARP_AGGREGATION_NONE;
    reduce_spec.length                      = (count/size);//reducce scatter 0
    reduce_spec.dtype                       = sharp_type;
    reduce_spec.root                        = 0;
    reduce_spec.op                          = op_type;


    ucc_info("***reduce datalen:%lu, ranksize:%d\n", reduce_spec.length, size);

    char *srcBufPtrInChar = (char *) args->src.info.buffer;
    char *dstBufPtrInChar = (char *) args->dst.info.buffer;

    ret = SHARP_COLL_SUCCESS;

    for(int rankCnt = 0; rankCnt < size; rankCnt++){

        if(rankCnt == 0)
            ret = sharp_coll_do_reduce_nb(team->sharp_comm, &reduce_spec, &task->req_handle); // TODO: change it to reduce_scatter
        else
            ret = sharp_coll_do_reduce_nb(team->sharp_comm, &reduce_spec, &sharp_reqs[rankCnt]);

        /*update src and dst ptr*/
        srcBufPtrInChar += offset;
        reduce_spec.sbuf_desc.buffer.ptr  = (void *)srcBufPtrInChar;

        dstBufPtrInChar += offset;
        reduce_spec.rbuf_desc.buffer.ptr = (void *)dstBufPtrInChar;

        /*update root*/
        reduce_spec.root += 1;

    }

    /*wait for all but first reduce to complete, and return the first of them to &task->req_handle, let later functions to test*/
    /*all but first sharp_reqs will be deallocated by sharp_coll_req_wait*/
    for(int rankCnt = 1; rankCnt < size; rankCnt++){

        ret = sharp_coll_req_wait(sharp_reqs[rankCnt]);
        if(ret != SHARP_COLL_SUCCESS){
            tl_error(UCC_TASK_LIB(task), "reduce scatter fail at rank:%d\n", rank);
            return UCC_ERR_LAST;
        }
    }


    if (ucc_unlikely(ret != SHARP_COLL_SUCCESS)) {
        tl_error(UCC_TASK_LIB(task), "reduce scatter REDUCE failed:%s",
                 sharp_coll_strerror(ret));
        coll_task->status = ucc_tl_sharp_status_to_ucc(ret);
        return ucc_task_complete(coll_task);
    }
    
    coll_task->status = UCC_INPROGRESS;
    //coll_task->status = UCC_OK;

    return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
    //return UCC_OK;
}
```

```c
ucc_status_t ucc_tl_sharp_reduce_scatter_init(ucc_tl_sharp_task_t *task)
{
    ucc_info("*********** sharp_reduce_scatter_init ************\n");
    ucc_coll_args_t *args = &TASK_ARGS(task);

    if (!ucc_coll_args_is_predefined_dt(args, UCC_RANK_INVALID)) {
        return UCC_ERR_NOT_SUPPORTED;
    }

    if ((!UCC_IS_INPLACE(*args) &&
         ucc_to_sharp_memtype[args->src.info.mem_type] == SHARP_MEM_TYPE_LAST) ||
        ucc_to_sharp_memtype[args->dst.info.mem_type] == SHARP_MEM_TYPE_LAST ||
        ucc_to_sharp_dtype[UCC_DT_PREDEFINED_ID(args->dst.info.datatype)] == SHARP_DTYPE_NULL ||
        ucc_to_sharp_reduce_op[args->op] == SHARP_OP_NULL) {
        return UCC_ERR_NOT_SUPPORTED;
    }

    task->super.post     = ucc_tl_sharp_reduce_scatter_start;
    task->super.progress = ucc_tl_sharp_collective_progress;
    return UCC_OK;
}
```

Other places that include changes are:

- Added function declaration in `tl_sharp_coll.h`

```c
ucc_status_t ucc_tl_sharp_reduce_scatter_init(ucc_tl_sharp_task_t *task);
```

- Added corresponding switch case for coll_init in `tl_sharp_context.c`

```c
case UCC_COLL_TYPE_REDUCE_SCATTER:
    tl_debug(UCC_TASK_LIB(task), "initing reduce scatter coll task %p", task);
    status = ucc_tl_sharp_reduce_scatter_init(task);
    break;
```
- in `tl_sharp.h`, we added enumeration for reduce_scatter
```c
#define UCC_TL_SHARP_SUPPORTED_COLLS                                           \
    (UCC_COLL_TYPE_ALLREDUCE | UCC_COLL_TYPE_BARRIER | UCC_COLL_TYPE_BCAST | UCC_COLL_TYPE_REDUCE_SCATTER | UCC_COLL_TYPE_REDUCE_SCATTERV)

``` 

### Performance Evaluation

The speedup of reduce scatter with SHARP is significant compared to UCP tl. The following table shows the performance of reduce scatter with SHARP and without SHARP. The performance is measured by the latency an calculated bandwidth of the reduce scatter operation.

```bash
# ########### perf sharp ON

#        Count        Size                Time, us                           Bandwidth, GB/s
#                                  avg         min         max         avg         max         min
#        16384       65536        9.54        8.80       10.84       20.62       22.33       18.15
#        32768      131072       14.96       14.38       16.42       26.28       27.34       23.95
#        65536      262144       25.60       24.75       27.57       30.72       31.77       28.53
#       131072      524288       48.41       47.42       50.96       32.49       33.17       30.86
#       262144     1048576       92.19       91.38       93.96       34.12       34.42       33.48
#       524288     2097152      181.14      180.35      183.06       34.73       34.88       34.37
#      1048576     4194304      359.04      358.27      360.94       35.05       35.12       34.86
#      2097152     8388608      758.77      757.90      760.94       33.17       33.20       33.07
#      4194304    16777216     1575.13     1574.16     1577.52       31.95       31.97       31.91
#      8388608    33554432     3164.77     3163.38     3166.61       31.81       31.82       31.79
#     16777216    67108864     6347.39     6346.47     6349.78       31.72       31.72       31.71
#     33554432   134217728    12681.29    12680.08    12683.31       31.75       31.75       31.75
#     67108864   268435456    25398.83    25397.85    25401.24       31.71       31.71       31.70
#    134217728   536870912    50849.42    50848.51    50852.04       31.67       31.67       31.67


# ########### perf sharp OFF (using UCP)

#        Count        Size                Time, us                           Bandwidth, GB/s
#                                  avg         min         max         avg         max         min
#        16384       65536       46.07       45.68       46.44        4.27        4.30        4.23
#        32768      131072       85.44       83.22       87.81        4.60        4.72        4.48
#        65536      262144      154.09      148.72      159.35        5.10        5.29        4.94
#       131072      524288      290.91      274.37      303.37        5.41        5.73        5.18
#       262144     1048576      554.53      530.35      576.88        5.67        5.93        5.45
#       524288     2097152     1093.23     1030.02     1144.22        5.75        6.11        5.50
#      1048576     4194304     2244.57     2147.11     2306.64        5.61        5.86        5.46
#      2097152     8388608     7092.01     7015.09     7175.36        3.55        3.59        3.51
#      4194304    16777216    25325.55    25221.17    25380.26        1.99        2.00        1.98
#      8388608    33554432    53533.08    52978.25    54052.87        1.88        1.90        1.86
#     16777216    67108864   107361.22   106633.57   108699.34        1.88        1.89        1.85
#     33554432   134217728   212698.12   210525.60   214735.00        1.89        1.91        1.88
#     67108864   268435456   422000.92   413962.71   426931.16        1.91        1.95        1.89
#    134217728   536870912   841231.09   827954.97   849455.64        1.91        1.95        1.90

```

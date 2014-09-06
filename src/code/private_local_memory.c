__kernel void private_local_memory(__global float* A, __global float* B, __global float* C, __local float* B_local)
{
    int x = get_global_id(0);
    int y, k, index;
    float acc;
    int rank = get_global_size(0);
    int iloc = get_local_id(0); 
    int nloc = get_local_size(0);

    float A_private[4096];

    for(index = 0; index < rank; index++) A_private[index] = A[x*rank + index];                                                      

    for (y=0; y < rank; y++) {
        for (k = iloc; k < rank; k += nloc) B_local[k] = B[k*rank+y];
        barrier(CLK_LOCAL_MEM_FENCE);

        acc  = 0.0f;
        for(index = 0; index <  rank; index++) acc +=  A_private[index] * B_local[index];
        C[x*rank + y] = acc;

        barrier(CLK_LOCAL_MEM_FENCE);
    }   
    return;
}


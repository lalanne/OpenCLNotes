__kernel void more_work(__global float* A, __global float* B, __global float* C)                                                     
{
    int x = get_global_id(0);
    int y, index;
    float acc;
    int rank = get_global_size(0);

    for (y=0; y < rank; y++) {
        acc  = 0.0f;
        for(index = 0; index <  rank; index++)
            acc +=  A[x*rank+index] * B[index*rank+y];
        C[x*rank + y] = acc;
    }   
    return;
}

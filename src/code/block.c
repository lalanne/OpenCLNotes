#define BLOCK_SIZE 512 
#define AS(i,j) as[j + i*BLOCK_SIZE]
#define BS(i,j) bs[j + i*BLOCK_SIZE]

__kernel __attribute__ 
void block(__global float* a, __global float* b, __global float* output, __local float* as, __local float* bs) 
{
    int block_x = get_group_id(0);
    int block_y = get_group_id(1);
    int thread_x = get_local_id(0);
    int thread_y = get_local_id(1);
    int rank = get_global_size(0);
    float running = 0.0f;

    int a_index = rank * BLOCK_SIZE * block_y;
    int b_index = BLOCK_SIZE * block_x;
    int a_step = BLOCK_SIZE;
    int b_step = BLOCK_SIZE * rank;

    int end = a_index + (rank - 1); 
    int r, c, n;

    for(r = a_index, c = b_index; r < end; r += a_step, c += b_step)
    {
        AS(thread_y, thread_x) = a[r + rank*thread_y + thread_x];
        BS(thread_y, thread_x) = b[c + rank*thread_y + thread_x];

        barrier(CLK_LOCAL_MEM_FENCE);

        for(n = 0; n < BLOCK_SIZE; ++n)
            running += AS(thread_y, n)*BS(n, thread_x);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    output[get_global_id(1)*get_global_size(0) + get_global_id(0)] = running;
}

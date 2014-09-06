__kernel void naive(__global float* A, __global float* B, __global float* C)
{
  int x = get_global_id(0);
  int y = get_global_id(1);
  int rank = get_global_size(0);
  float acc = 0.0f;

  for (int i=0; i<rank; i++) {
    int aIndex = x*rank + i;
    int bIndex = i*rank + y;
    acc +=  A[aIndex] * B[bIndex];
  }
  
  C[x*rank + y] = acc;
  return;
}


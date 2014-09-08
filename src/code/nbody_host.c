globalworkitem_dim0= 256; 
globalworkitem_dim1=120; 
localworkitem_dim0= 256; 
localworkitem_dim1=1; 
globalWorkSize[2]={ globalworkitem_dim0, globalworkitem_dim1}; 
localWorkSize[2]={ localworkitem_dim0, localworkitem_dim1};

for( i=0; i<Num_of_Vertices; i+=(globalworkitem_dim0)*(globalworkitem_dim1)) {
    kernel(); 
}

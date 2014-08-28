 %temp24vector_func.i = insertelement <16 x i32> undef, i32 %36, i32 0                                   
 %vector23vector_func.i = shufflevector <16 x i32> %temp24vector_func.i, <16 x i32> undef, <16 x i32> zeroinitializer
 %37 = add nsw <16 x i32> %24, %vector23vector_func.i
 %extract.05.i = extractelement <16 x i32> %37, i32 0

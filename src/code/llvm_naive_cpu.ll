%temp15vector_func.i = insertelement <4 x i32> undef, i32 %27, i32 0
%vector14vector_func.i = shufflevector <4 x i32> %temp15vector_func.i, <4 x i32> undef, <4 x i32> zeroinitializer
%28 = add nsw <4 x i32> %24, %vector14vector_func.i                
%29 = sext <4 x i32> %28 to <4 x i64>

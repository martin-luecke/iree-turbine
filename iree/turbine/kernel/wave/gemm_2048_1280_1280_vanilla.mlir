#translation = #iree_codegen.translation_info<pipeline = None workgroup_size = [128, 2, 1] subgroup_size = 64>
module attributes {transform.with_named_sequence} {
  stream.executable private @gemm {
    stream.executable.export public @gemm workgroups() -> (index, index, index) {
      %c32 = arith.constant 32 : index
      %c20 = arith.constant 20 : index
      %c1 = arith.constant 1 : index
      stream.return %c32, %c20, %c1 : index, index, index
    }
    builtin.module {
      func.func @gemm(%_A: !stream.binding, %_B: !stream.binding, %_C: !stream.binding) attributes {translation_info = #translation} {
        %c19 = arith.constant 19 : index
        %c17 = arith.constant 17 : index
        %c3 = arith.constant 3 : index
        %c2 = arith.constant 2 : index
        %c128 = arith.constant 128 : index
        %c18 = arith.constant 18 : index
        %c48 = arith.constant 48 : index
        %c4 = arith.constant 4 : index
        %c8 = arith.constant 8 : index
        %c1 = arith.constant 1 : index
        %c32 = arith.constant 32 : index
        %c16 = arith.constant 16 : index
        %c64 = arith.constant 64 : index
        %c0 = arith.constant 0 : index
        %cst = arith.constant dense<0.000000e+00> : vector<4xf32>
        %workgroup_id_0 = stream.dispatch.workgroup.id[0] : index
        %workgroup_id_1 = stream.dispatch.workgroup.id[1] : index
        %thread_id_x = gpu.thread_id  x
        %thread_id_y = gpu.thread_id  y
        %thread_id_z = gpu.thread_id  z
        %A_shared = memref.alloc() : memref<64x68xf16, #gpu.address_space<workgroup>>
        %B_shared = memref.alloc() : memref<64x68xf16, #gpu.address_space<workgroup>>
        %A = stream.binding.subspan %_A[%c0] : !stream.binding -> memref<2048x1280xf16, strided<[1280, 1], offset: ?>>
        %1 = arith.muli %workgroup_id_0, %c64 overflow<nsw, nuw> : index
        %2 = arith.muli %thread_id_y, %c16 overflow<nsw, nuw> : index
        %3 = arith.muli %thread_id_z, %c32 overflow<nsw, nuw> : index
        %4 = arith.divsi %thread_id_x, %c8 : index
        %5 = arith.addi %4, %3 overflow<nsw, nuw> : index
        %6 = arith.addi %5, %2 overflow<nsw, nuw> : index
        %7 = arith.remsi %6, %c64 : index
        %8 = arith.addi %7, %1 overflow<nsw, nuw> : index
        %9 = arith.remsi %thread_id_x, %c8 : index
        %10 = arith.muli %9, %c8 overflow<nsw, nuw> : index
        %11 = vector.load %A[%8, %10] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<8xf16>
        %12 = arith.addi %6, %c32 overflow<nsw, nuw> : index
        %13 = arith.remsi %12, %c64 : index
        %14 = arith.addi %13, %1 overflow<nsw, nuw> : index
        %15 = vector.load %A[%14, %10] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<8xf16>
        %B = stream.binding.subspan %_B[%c0] : !stream.binding -> memref<1280x1280xf16, strided<[1280, 1], offset: ?>>
        %17 = arith.muli %workgroup_id_1, %c64 overflow<nsw, nuw> : index
        %18 = arith.addi %7, %17 overflow<nsw, nuw> : index
        %19 = vector.load %B[%18, %10] : memref<1280x1280xf16, strided<[1280, 1], offset: ?>>, vector<8xf16>
        %20 = arith.addi %13, %17 overflow<nsw, nuw> : index
        %21 = vector.load %B[%20, %10] : memref<1280x1280xf16, strided<[1280, 1], offset: ?>>, vector<8xf16>
        vector.store %11, %A_shared[%7, %10] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
        vector.store %15, %A_shared[%13, %10] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
        vector.store %19, %B_shared[%7, %10] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
        vector.store %21, %B_shared[%13, %10] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
        amdgpu.lds_barrier
        %22 = arith.divsi %thread_id_x, %c64 : index
        %23 = arith.muli %22, %c32 overflow<nsw, nuw> : index
        %24 = arith.remsi %thread_id_x, %c16 : index
        %25 = arith.addi %24, %23 overflow<nsw, nuw> : index
        %26 = arith.addi %25, %c16 overflow<nsw, nuw> : index
        %27 = arith.remsi %thread_id_x, %c64 : index
        %28 = arith.divsi %27, %c16 : index
        %29 = arith.muli %28, %c4 overflow<nsw, nuw> : index
        %30 = arith.addi %29, %c32 overflow<nsw, nuw> : index
        %31 = vector.load %A_shared[%26, %30] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %32 = arith.addi %29, %c48 overflow<nsw, nuw> : index
        %33 = vector.load %A_shared[%26, %32] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %34 = arith.muli %thread_id_y, %c32 overflow<nsw, nuw> : index
        %35 = arith.addi %24, %34 overflow<nsw, nuw> : index
        %36 = arith.addi %35, %c16 overflow<nsw, nuw> : index
        %37 = vector.load %B_shared[%36, %30] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %38 = vector.load %B_shared[%36, %32] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %39 = vector.load %A_shared[%26, %29] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %40 = arith.addi %29, %c16 overflow<nsw, nuw> : index
        %41 = vector.load %A_shared[%26, %40] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %42 = vector.load %B_shared[%36, %29] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %43 = vector.load %B_shared[%36, %40] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %44 = vector.load %B_shared[%35, %29] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %45 = vector.load %B_shared[%35, %40] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %46 = vector.load %B_shared[%35, %30] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %47 = vector.load %B_shared[%35, %32] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %48 = amdgpu.mfma %39 * %42 + %cst {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %49 = arith.addi %10, %c64 overflow<nsw, nuw> : index
        %50 = vector.load %A[%8, %49] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<8xf16>
        %51 = vector.load %A[%14, %49] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<8xf16>
        %52 = vector.load %B[%18, %49] : memref<1280x1280xf16, strided<[1280, 1], offset: ?>>, vector<8xf16>
        %53 = vector.load %B[%20, %49] : memref<1280x1280xf16, strided<[1280, 1], offset: ?>>, vector<8xf16>
        %54 = vector.load %A_shared[%25, %29] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %55 = vector.load %A_shared[%25, %40] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %56 = vector.load %A_shared[%25, %30] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %57 = vector.load %A_shared[%25, %32] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %58 = amdgpu.mfma %39 * %44 + %cst {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %59 = amdgpu.mfma %41 * %43 + %48 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %60 = amdgpu.mfma %54 * %44 + %cst {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %61 = amdgpu.mfma %54 * %42 + %cst {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %62 = amdgpu.mfma %41 * %45 + %58 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %63 = amdgpu.mfma %31 * %37 + %59 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        amdgpu.lds_barrier
        vector.store %50, %A_shared[%7, %10] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
        vector.store %51, %A_shared[%13, %10] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
        vector.store %52, %B_shared[%7, %10] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
        vector.store %53, %B_shared[%13, %10] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
        %64 = amdgpu.mfma %55 * %45 + %60 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %65 = amdgpu.mfma %55 * %43 + %61 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %66 = amdgpu.mfma %31 * %46 + %62 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %67 = amdgpu.mfma %33 * %38 + %63 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        amdgpu.lds_barrier
        %68 = vector.load %A_shared[%26, %30] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %69 = vector.load %A_shared[%26, %32] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %70 = vector.load %B_shared[%36, %30] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %71 = vector.load %B_shared[%36, %32] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %72 = amdgpu.mfma %56 * %46 + %64 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %73 = amdgpu.mfma %56 * %37 + %65 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %74 = amdgpu.mfma %33 * %47 + %66 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %75 = vector.load %A_shared[%26, %29] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %76 = vector.load %A_shared[%26, %40] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %77 = vector.load %B_shared[%36, %29] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %78 = vector.load %B_shared[%36, %40] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %79:15 = scf.for %arg3 = %c0 to %c18 step %c1 iter_args(%arg4 = %74, %arg5 = %67, %arg6 = %57, %arg7 = %75, %arg8 = %76, %arg9 = %68, %arg10 = %69, %arg11 = %47, %arg12 = %77, %arg13 = %78, %arg14 = %70, %arg15 = %38, %arg16 = %71, %arg17 = %72, %arg18 = %73) -> (vector<4xf32>, vector<4xf32>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf32>, vector<4xf32>) {
          // I now can pass the buffers and just need to switch them at the end in the yield.
          // Need to find out where to have the compute and where to have the next
          %135 = amdgpu.mfma %arg6 * %arg11 + %arg17 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %136 = amdgpu.mfma %arg6 * %arg15 + %arg18 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %137 = vector.load %B_shared[%35, %29] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %138 = vector.load %B_shared[%35, %40] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %139 = vector.load %B_shared[%35, %30] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %140 = vector.load %B_shared[%35, %32] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %141 = amdgpu.mfma %arg7 * %arg12 + %arg5 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %142 = arith.muli %arg3, %c64 overflow<nsw, nuw> : index
          %143 = arith.addi %142, %10 overflow<nsw, nuw> : index
          %144 = arith.addi %143, %c128 overflow<nsw, nuw> : index
          %145 = vector.load %A[%8, %144] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<8xf16>
          %146 = vector.load %A[%14, %144] : memref<2048x1280xf16, strided<[1280, 1], offset: ?>>, vector<8xf16>
          %147 = vector.load %B[%18, %144] : memref<1280x1280xf16, strided<[1280, 1], offset: ?>>, vector<8xf16>
          %148 = vector.load %B[%20, %144] : memref<1280x1280xf16, strided<[1280, 1], offset: ?>>, vector<8xf16>
          %149 = vector.load %A_shared[%25, %29] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %150 = vector.load %A_shared[%25, %40] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %151 = vector.load %A_shared[%25, %30] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %152 = vector.load %A_shared[%25, %32] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %153 = amdgpu.mfma %arg7 * %137 + %arg4 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %154 = amdgpu.mfma %arg8 * %arg13 + %141 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %155 = amdgpu.mfma %149 * %137 + %135 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %156 = amdgpu.mfma %149 * %arg12 + %136 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %157 = amdgpu.mfma %arg8 * %138 + %153 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %158 = amdgpu.mfma %arg9 * %arg14 + %154 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          amdgpu.lds_barrier
          vector.store %145, %A_shared[%7, %10] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
          vector.store %146, %A_shared[%13, %10] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
          vector.store %147, %B_shared[%7, %10] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
          vector.store %148, %B_shared[%13, %10] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<8xf16>
          %159 = amdgpu.mfma %150 * %138 + %155 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %160 = amdgpu.mfma %150 * %arg13 + %156 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %161 = amdgpu.mfma %arg9 * %139 + %157 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %162 = amdgpu.mfma %arg10 * %arg16 + %158 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          amdgpu.lds_barrier
          %163 = vector.load %A_shared[%26, %30] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %164 = vector.load %A_shared[%26, %32] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %165 = vector.load %B_shared[%36, %30] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %166 = vector.load %B_shared[%36, %32] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %167 = amdgpu.mfma %151 * %139 + %159 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %168 = amdgpu.mfma %151 * %arg14 + %160 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %169 = amdgpu.mfma %arg10 * %140 + %161 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
          %170 = vector.load %A_shared[%26, %29] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %171 = vector.load %A_shared[%26, %40] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %172 = vector.load %B_shared[%36, %29] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          %173 = vector.load %B_shared[%36, %40] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
          scf.yield %169, %162, %152, %170, %171, %163, %164, %140, %172, %173, %165, %arg16, %166, %167, %168 : vector<4xf32>, vector<4xf32>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf16>, vector<4xf32>, vector<4xf32>
        }
        %80 = amdgpu.mfma %79#2 * %79#7 + %79#13 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %81 = amdgpu.mfma %79#2 * %79#11 + %79#14 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %82 = vector.load %B_shared[%35, %29] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %83 = vector.load %B_shared[%35, %40] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %84 = vector.load %B_shared[%35, %30] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %85 = vector.load %B_shared[%35, %32] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %86 = amdgpu.mfma %79#3 * %79#8 + %79#1 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %87 = vector.load %A_shared[%25, %29] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %88 = vector.load %A_shared[%25, %40] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %89 = vector.load %A_shared[%25, %30] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %90 = vector.load %A_shared[%25, %32] : memref<64x68xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        %91 = amdgpu.mfma %79#3 * %82 + %79#0 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %92 = amdgpu.mfma %79#4 * %79#9 + %86 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %93 = amdgpu.mfma %87 * %82 + %80 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %94 = amdgpu.mfma %87 * %79#8 + %81 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %95 = amdgpu.mfma %79#4 * %83 + %91 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %96 = amdgpu.mfma %79#5 * %79#10 + %92 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %97 = amdgpu.mfma %88 * %83 + %93 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %98 = amdgpu.mfma %88 * %79#9 + %94 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %99 = amdgpu.mfma %79#5 * %84 + %95 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %100 = amdgpu.mfma %79#6 * %79#12 + %96 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %101 = amdgpu.mfma %89 * %84 + %97 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %102 = amdgpu.mfma %89 * %79#10 + %98 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %103 = amdgpu.mfma %79#6 * %85 + %99 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %104 = amdgpu.mfma %90 * %85 + %101 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %105 = amdgpu.mfma %90 * %79#12 + %102 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        %106 = vector.extract_strided_slice %104 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %C = stream.binding.subspan %_C[%c0] : !stream.binding -> memref<2048x1280xf32, strided<[1280, 1], offset: ?>>
        %108 = arith.addi %1, %23 overflow<nsw, nuw> : index
        %109 = arith.addi %108, %29 overflow<nsw, nuw> : index
        %110 = arith.addi %24, %17 overflow<nsw, nuw> : index
        %111 = arith.addi %110, %34 overflow<nsw, nuw> : index
        vector.store %106, %C[%109, %111] : memref<2048x1280xf32, strided<[1280, 1], offset: ?>>, vector<1xf32>
        %112 = vector.extract_strided_slice %104 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %113 = arith.addi %109, %c1 overflow<nsw, nuw> : index
        vector.store %112, %C[%113, %111] : memref<2048x1280xf32, strided<[1280, 1], offset: ?>>, vector<1xf32>
        %114 = vector.extract_strided_slice %104 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %115 = arith.addi %109, %c2 overflow<nsw, nuw> : index
        vector.store %114, %C[%115, %111] : memref<2048x1280xf32, strided<[1280, 1], offset: ?>>, vector<1xf32>
        %116 = vector.extract_strided_slice %104 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %117 = arith.addi %109, %c3 overflow<nsw, nuw> : index
        vector.store %116, %C[%117, %111] : memref<2048x1280xf32, strided<[1280, 1], offset: ?>>, vector<1xf32>
        %118 = vector.extract_strided_slice %105 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %119 = arith.addi %111, %c16 overflow<nsw, nuw> : index
        vector.store %118, %C[%109, %119] : memref<2048x1280xf32, strided<[1280, 1], offset: ?>>, vector<1xf32>
        %120 = vector.extract_strided_slice %105 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %120, %C[%113, %119] : memref<2048x1280xf32, strided<[1280, 1], offset: ?>>, vector<1xf32>
        %121 = vector.extract_strided_slice %105 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %121, %C[%115, %119] : memref<2048x1280xf32, strided<[1280, 1], offset: ?>>, vector<1xf32>
        %122 = vector.extract_strided_slice %105 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %122, %C[%117, %119] : memref<2048x1280xf32, strided<[1280, 1], offset: ?>>, vector<1xf32>
        %123 = vector.extract_strided_slice %103 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %124 = arith.addi %109, %c16 overflow<nsw, nuw> : index
        vector.store %123, %C[%124, %111] : memref<2048x1280xf32, strided<[1280, 1], offset: ?>>, vector<1xf32>
        %125 = vector.extract_strided_slice %103 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %126 = arith.addi %109, %c17 overflow<nsw, nuw> : index
        vector.store %125, %C[%126, %111] : memref<2048x1280xf32, strided<[1280, 1], offset: ?>>, vector<1xf32>
        %127 = vector.extract_strided_slice %103 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %128 = arith.addi %109, %c18 overflow<nsw, nuw> : index
        vector.store %127, %C[%128, %111] : memref<2048x1280xf32, strided<[1280, 1], offset: ?>>, vector<1xf32>
        %129 = vector.extract_strided_slice %103 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %130 = arith.addi %109, %c19 overflow<nsw, nuw> : index
        vector.store %129, %C[%130, %111] : memref<2048x1280xf32, strided<[1280, 1], offset: ?>>, vector<1xf32>
        %131 = vector.extract_strided_slice %100 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %131, %C[%124, %119] : memref<2048x1280xf32, strided<[1280, 1], offset: ?>>, vector<1xf32>
        %132 = vector.extract_strided_slice %100 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %132, %C[%126, %119] : memref<2048x1280xf32, strided<[1280, 1], offset: ?>>, vector<1xf32>
        %133 = vector.extract_strided_slice %100 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %133, %C[%128, %119] : memref<2048x1280xf32, strided<[1280, 1], offset: ?>>, vector<1xf32>
        %134 = vector.extract_strided_slice %100 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %134, %C[%130, %119] : memref<2048x1280xf32, strided<[1280, 1], offset: ?>>, vector<1xf32>
        return
      }
    }
  }
  func.func @isolated_benchmark(%arg0: tensor<2048x1280xf16>, %arg1: tensor<1280x1280xf16>, %arg2: tensor<2048x1280xf32>) -> tensor<2048x1280xf32> {
    %0 = flow.dispatch @gemm::@gemm(%arg0, %arg1, %arg2) : (tensor<2048x1280xf16>, tensor<1280x1280xf16>, tensor<2048x1280xf32>) -> %arg2
    return %0 : tensor<2048x1280xf32>
  }
}
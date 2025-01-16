# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Any, Callable, Optional
import torch.fx as fx
import inspect

from .symbolic_constraints import SymbolicAlias

from ..compiler import builder, dispatch_codegen, kernel_codegen, host_codegen
from ..compiler.ir import Context, Operation, builtin_d
from .codegen import WaveEmitter
from .constraints import (
    Constraint,
    HardwareConstraint,
    TilingConstraint,
    WaveConstraint,
    WorkgroupConstraint,
    get_grid_shape,
)
from .codegen import WaveEmitter
from .expansion.expansion import expand_graph
from .promotion import promote_placeholders
from .hoisting import hoist_loop_invariant_ops
from .utils import (
    canonicalize_module,
    compile_to_vmfb,
    invoke_vmfb,
    safe_subs,
    remove_chained_getresult,
    remove_chained_extractslice,
    subs_idxc,
    delinearize_index,
    _write_file,
    initialize_iter_args,
    print_graph,
    print_subgraph,
    print_trace
)
from .minimize_global_loads import minimize_global_loads
from .decompose_reduce_ops import decompose_reduce_ops
from .decompose_vmma_ops import decompose_vmma_ops
from .barriers import add_shared_memory_barriers
from ..lang import Grid, IndexMapping
from ..lang.global_symbols import *
from ..ops import wave_ops
from ..ops.wave_ops import Reduction, CustomOp, get_custom, IterArg
from .index_sequence_analysis import (
    partition_ops_with_gpr_offsets,
    partition_strided_operators,
    set_node_indices,
    set_post_expansion_indices,
)
from .shared_memory_indexing import (
    apply_shared_memory_indexing_corrections,
    align_index_sizes,
)
from .scheduling.schedule import schedule_graph
from .._support.indexing import IndexingContext, IndexExpr
from .type_inference import infer_types
import iree.turbine.kernel.lang as tkl
from .._support.tracing import (
    CapturedTrace,
    CompiledContext,
    KernelRegionGraph,
    Launchable,
)
from .cache import is_cache_enabled, get_cache_manager, invoke_cached_kernel

import sympy

__all__ = ["wave", "wave_trace_only"]

inject_custom_kernel = True
custom_kernel = """#translation = #iree_codegen.translation_info<pipeline = None workgroup_size = [128, 2, 1] subgroup_size = 64>
module attributes {transform.with_named_sequence} {
  stream.executable private @gemm {
    stream.executable.export public @gemm workgroups() -> (index, index, index) {
      %c32 = arith.constant 32 : index
      %c20 = arith.constant 20 : index
      %c1 = arith.constant 1 : index
      stream.return %c32, %c20, %c1 : index, index, index
    }
    builtin.module {
      func.func @gemm(%A1: !stream.binding, %B1: !stream.binding, %C: !stream.binding) attributes {translation_info = #translation} {
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
        %A = stream.binding.subspan %A1[%c0] : !stream.binding -> memref<2048x1280xf16, strided<[1280, 1], offset: ?>>
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
        %B = stream.binding.subspan %B1[%c0] : !stream.binding -> memref<1280x1280xf16, strided<[1280, 1], offset: ?>>
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
        %107 = stream.binding.subspan %C[%c0] : !stream.binding -> memref<2048x1280xf32, strided<[1280, 1], offset: ?>>
        %108 = arith.addi %1, %23 overflow<nsw, nuw> : index
        %109 = arith.addi %108, %29 overflow<nsw, nuw> : index
        %110 = arith.addi %24, %17 overflow<nsw, nuw> : index
        %111 = arith.addi %110, %34 overflow<nsw, nuw> : index
        vector.store %106, %107[%109, %111] : memref<2048x1280xf32, strided<[1280, 1], offset: ?>>, vector<1xf32>
        %112 = vector.extract_strided_slice %104 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %113 = arith.addi %109, %c1 overflow<nsw, nuw> : index
        vector.store %112, %107[%113, %111] : memref<2048x1280xf32, strided<[1280, 1], offset: ?>>, vector<1xf32>
        %114 = vector.extract_strided_slice %104 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %115 = arith.addi %109, %c2 overflow<nsw, nuw> : index
        vector.store %114, %107[%115, %111] : memref<2048x1280xf32, strided<[1280, 1], offset: ?>>, vector<1xf32>
        %116 = vector.extract_strided_slice %104 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %117 = arith.addi %109, %c3 overflow<nsw, nuw> : index
        vector.store %116, %107[%117, %111] : memref<2048x1280xf32, strided<[1280, 1], offset: ?>>, vector<1xf32>
        %118 = vector.extract_strided_slice %105 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %119 = arith.addi %111, %c16 overflow<nsw, nuw> : index
        vector.store %118, %107[%109, %119] : memref<2048x1280xf32, strided<[1280, 1], offset: ?>>, vector<1xf32>
        %120 = vector.extract_strided_slice %105 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %120, %107[%113, %119] : memref<2048x1280xf32, strided<[1280, 1], offset: ?>>, vector<1xf32>
        %121 = vector.extract_strided_slice %105 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %121, %107[%115, %119] : memref<2048x1280xf32, strided<[1280, 1], offset: ?>>, vector<1xf32>
        %122 = vector.extract_strided_slice %105 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %122, %107[%117, %119] : memref<2048x1280xf32, strided<[1280, 1], offset: ?>>, vector<1xf32>
        %123 = vector.extract_strided_slice %103 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %124 = arith.addi %109, %c16 overflow<nsw, nuw> : index
        vector.store %123, %107[%124, %111] : memref<2048x1280xf32, strided<[1280, 1], offset: ?>>, vector<1xf32>
        %125 = vector.extract_strided_slice %103 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %126 = arith.addi %109, %c17 overflow<nsw, nuw> : index
        vector.store %125, %107[%126, %111] : memref<2048x1280xf32, strided<[1280, 1], offset: ?>>, vector<1xf32>
        %127 = vector.extract_strided_slice %103 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %128 = arith.addi %109, %c18 overflow<nsw, nuw> : index
        vector.store %127, %107[%128, %111] : memref<2048x1280xf32, strided<[1280, 1], offset: ?>>, vector<1xf32>
        %129 = vector.extract_strided_slice %103 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        %130 = arith.addi %109, %c19 overflow<nsw, nuw> : index
        vector.store %129, %107[%130, %111] : memref<2048x1280xf32, strided<[1280, 1], offset: ?>>, vector<1xf32>
        %131 = vector.extract_strided_slice %100 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %131, %107[%124, %119] : memref<2048x1280xf32, strided<[1280, 1], offset: ?>>, vector<1xf32>
        %132 = vector.extract_strided_slice %100 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %132, %107[%126, %119] : memref<2048x1280xf32, strided<[1280, 1], offset: ?>>, vector<1xf32>
        %133 = vector.extract_strided_slice %100 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %133, %107[%128, %119] : memref<2048x1280xf32, strided<[1280, 1], offset: ?>>, vector<1xf32>
        %134 = vector.extract_strided_slice %100 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
        vector.store %134, %107[%130, %119] : memref<2048x1280xf32, strided<[1280, 1], offset: ?>>, vector<1xf32>
        return
      }
    }
  }
  func.func @isolated_benchmark(%arg0: tensor<2048x1280xf16>, %arg1: tensor<1280x1280xf16>, %arg2: tensor<2048x1280xf32>) -> tensor<2048x1280xf32> {
    %0 = flow.dispatch @gemm::@gemm(%arg0, %arg1, %arg2) : (tensor<2048x1280xf16>, tensor<1280x1280xf16>, tensor<2048x1280xf32>) -> %arg2
    return %0 : tensor<2048x1280xf32>
  }
}
"""

def wave(constraints: Optional[list[Constraint]] = None):
    def decorator(f: Callable[..., Any]) -> "LaunchableWave":
        return LaunchableWave(constraints, f.__name__, f)

    return decorator


def wave_trace_only(constraints: Optional[list[Constraint]] = None):
    def decorator(f: Callable[..., Any]) -> "Callable[[], CapturedTrace]":
        wave = LaunchableWave(constraints, f.__name__, f)
        return wave._trace  # type: ignore

    return decorator


class LaunchableWave(Launchable):
    def __init__(
        self,
        constraints: Optional[list[Constraint]],
        name: str,
        eager_function: Callable[[Any], Any],
    ):
        super().__init__(eager_function)

        self.constraints = constraints if constraints else []
        self.induction_vars: dict[CustomOp, IndexExpr] = {}
        self._name = name
        self._f = eager_function
        self._sig = inspect.signature(eager_function)

        self.grid_type = Grid[tuple(get_grid_shape(self.workgroup_constraints))]

    @property
    def workgroup_constraints(self) -> list[WorkgroupConstraint]:
        return [
            constraint
            for constraint in self.constraints
            if isinstance(constraint, WorkgroupConstraint)
        ]

    @property
    def tiling_constraints(self) -> list[TilingConstraint]:
        return [
            constraint
            for constraint in self.constraints
            if isinstance(constraint, TilingConstraint)
        ]

    @property
    def wave_constraints(self) -> list[WaveConstraint]:
        return [
            constraint
            for constraint in self.constraints
            if isinstance(constraint, WaveConstraint)
        ]

    @property
    def hardware_constraints(self) -> list[HardwareConstraint]:
        return [
            constraint
            for constraint in self.constraints
            if isinstance(constraint, HardwareConstraint)
        ]

    @property
    def symbolic_constraints(self) -> list[HardwareConstraint]:
        return [
            constraint
            for constraint in self.constraints
            if isinstance(constraint, SymbolicAlias)
        ]

    def _trace(self) -> CapturedTrace:
        region_graph = KernelRegionGraph()
        with CompiledContext(region_graph, grid_type=self.grid_type) as context:
            # Get all explictly defined custom ops
            custom_ops: dict[str, wave_ops.CustomOp] = {
                cls.tkw_op_name: cls
                for _, cls in inspect.getmembers(wave_ops, inspect.isclass)
                if issubclass(cls, wave_ops.CustomOp) and hasattr(cls, "tkw_op_name")
            }

            # Register custom ops
            for name, op in custom_ops.items():
                context.register_custom_op(name, op)

            with region_graph.subtracer() as subtracer:
                root_name, _ = subtracer.trace(self._f)
                trace = CapturedTrace(region_graph, root_name)

        return trace

    def create_induction_vars(self, trace: CapturedTrace) -> None:
        """
        Creates induction variables for all the reductions in the graph
        and associates tiling constraints all the reduction dimensions
        with the appropriate induction variables.

        """

        def is_reduction(node: fx.Node):
            custom = get_custom(node)
            return isinstance(custom, Reduction)

        reduction_nodes = trace.walk(is_reduction)
        for node in reduction_nodes:
            custom = get_custom(node)
            self.induction_vars[custom] = tkl.IndexSymbol(
                "$ARG" + str(custom.axis), integer=True, nonnegative=True
            )
            for tiling_constraint in self.tiling_constraints:
                if tiling_constraint.dim == custom.axis:
                    tiling_constraint.induction_var = self.induction_vars[custom]

    def initialize_wave_constraints(self, trace: CapturedTrace) -> None:
        """
        For each wave constraint, determines the appropriate wave id by looking
        for workgroup constraints along the same dimension and using information
        from the hardware constraints.

        """

        hardware_constraint = self.hardware_constraints[0]
        for wave_constraint in self.wave_constraints:
            for workgroup_constraint in self.workgroup_constraints:
                # The wave_id is the same as the thread_id, with the exception
                # of wave_id[0] = thread_id[0] / threads_per_wave. This is
                # a convention that we adopt.
                if wave_constraint.dim == workgroup_constraint.dim:
                    wave_constraint.wave_id = (
                        hardware_constraint.get_thread_id_from_workgroup_dim(
                            workgroup_constraint.workgroup_dim
                        )
                    )
                    if workgroup_constraint.workgroup_dim == 0:
                        wave_constraint.wave_id = sympy.floor(
                            wave_constraint.wave_id
                            / hardware_constraint.threads_per_wave
                        )

    def initialize_reductions(self, trace: CapturedTrace) -> None:
        """
        For each reduction, initializes the reduction count by looking at the
        tiling constraints associated with the reduction.

        """
        is_reduction = lambda node: isinstance(get_custom(node), Reduction)
        for reduction in trace.walk(is_reduction):
            for tiling_constraint in self.tiling_constraints:
                if tiling_constraint.dim == get_custom(reduction).axis:
                    reduction.count = subs_idxc(tiling_constraint.count)

    def get_workgroup_dims(self) -> list[int]:
        """
        Returns the workgroup dimensions that are not aliased.
        """
        # Ignore aliased variables. They will be handled separately.
        aliased_dims = [
            x.source for x in self.constraints if isinstance(x, SymbolicAlias)
        ]
        workgroup_dims = {
            x.workgroup_dim: x
            for x in self.workgroup_constraints
            if x.dim not in aliased_dims
        }
        return workgroup_dims

    def update_aliased_workgroup_constraints(
        self, workgroup_dims: dict[int, int]
    ) -> None:
        """
        This function updates the wg_dim for aliased workgroup constraints.
        """
        aliased_dims = [
            x.source for x in self.constraints if isinstance(x, SymbolicAlias)
        ]
        # Update the workgroup constraints for aliases sources.
        for constraint in self.workgroup_constraints:
            if constraint.dim in aliased_dims:
                constraint.wg_dim = workgroup_dims[constraint.workgroup_dim].wg_dim

    def initialize_workgroup_constraints(self, trace: CapturedTrace) -> None:
        """
        For kernels that distribute more than three dimensions among workgroups,
        we need to update the workgroup constraints for dimensions >= 2
        with the appropriate workgroup index.
        """

        workgroup_dims = self.get_workgroup_dims()
        if all(x <= 2 for x in workgroup_dims.keys()):
            return
        shape = [
            subs_idxc(workgroup_dims[i].count)
            for i in range(2, max(workgroup_dims.keys()) + 1)
        ]
        new_workgroup_dims = delinearize_index(WORKGROUP_2, shape)
        for i in range(2, max(workgroup_dims.keys()) + 1):
            workgroup_dims[i].wg_dim = new_workgroup_dims[i - 2]
        self.update_aliased_workgroup_constraints(workgroup_dims)

    def initialize_symbolic_constraints(self, trace: CapturedTrace) -> None:
        """
        For each symbolic constraint, create new constraints for the
        related symbolic values with appropriate substitutions.
        """
        new_wg_constraints, new_wave_constraints, new_tiling_constraints = [], [], []
        for symbolic_constraint in self.symbolic_constraints:
            new_wg_constraints += symbolic_constraint.create_new_constraints(
                self.workgroup_constraints
            )
            new_wave_constraints += symbolic_constraint.create_new_constraints(
                self.wave_constraints
            )
            new_tiling_constraints += symbolic_constraint.create_new_constraints(
                self.tiling_constraints
            )
        # Remove wave constraints with same tile size as workgroup constraints
        for wave_constraint in new_wave_constraints:
            for workgroup_constraint in new_wg_constraints:
                if (
                    wave_constraint.dim == workgroup_constraint.dim
                    and wave_constraint.tile_size == workgroup_constraint.tile_size
                ):
                    new_wave_constraints.remove(wave_constraint)
        self.constraints += (
            new_wg_constraints + new_wave_constraints + new_tiling_constraints
        )
        idxc = IndexingContext.current()
        for constraint in self.symbolic_constraints:
            if subs_idxc(constraint.target).is_number:
                idxc._bind_symbol(
                    constraint.source,
                    subs_idxc(constraint.source_to_target(constraint.target)),
                )

    def _trace_and_get_kernel_signature(
        self,
        args,
        kwargs,
        context: Optional[Context] = None,
        module_op: Optional[Operation] = None,
    ) -> CapturedTrace:
        # Trace the function.
        graph = self._trace()

        initialize_iter_args(graph)
        self.create_induction_vars(graph)
        self.initialize_wave_constraints(graph)
        self.initialize_reductions(graph)
        self.initialize_symbolic_constraints(graph)
        self.initialize_workgroup_constraints(graph)

        idxc = IndexingContext.current()
        idxc.finalize()

        # Initialize Vector shapes
        self.hardware_constraints[0].subs_vector_shapes(idxc.subs)

        # Do type inference.
        infer_types(graph)

        # Promote the placeholders to the appropriate address space.
        promote_placeholders(graph, self.constraints)

        # Set indices.
        set_node_indices(graph, self.constraints)

        # Expansion
        expand_graph(graph, self.constraints)

        # Set post expansion indices.
        set_post_expansion_indices(graph, self.constraints)

        # Clean up chains of GetResults
        remove_chained_getresult(graph)

        # Optimizations.
        decompose_vmma_ops(graph, self.constraints)
        hoist_loop_invariant_ops(graph, self.constraints)
        minimize_global_loads(graph, self.constraints)

        # Apply shared memory indexing corrections.
        apply_shared_memory_indexing_corrections(graph, self.constraints)

        # Partition strided operators.
        partition_ops_with_gpr_offsets(graph, self.constraints)
        partition_strided_operators(graph, self.constraints)
        remove_chained_extractslice(graph)

        # Decompose reduce Ops.
        decompose_reduce_ops(graph, self.constraints, idxc.subs)

        # Schedule the reduction ops.
        # Scheduling should always be used with use_scheduling_barriers=True,
        # as this is the only way we can ensure that LLVM enforces our desired schedule.
        # However, due a bug in LLVM, you will need to patch your local LLVM repo
        # with the following commit: https://github.com/kerbowa/llvm-project/commit/ee52732cddae42deed2e3387a83b20ec05860b4e
        # Specifically:
        # git fetch https://github.com/kerbowa/llvm-project.git ee52732cddae42deed2e3387a83b20ec05860b4e
        # git cherry-pick ee52732cddae42deed2e3387a83b20ec05860b4e
        # [Manually resolve conflicts consistent with the PR]
        if kwargs.get("schedule", False):
            use_scheduling_barriers = kwargs.get("use_scheduling_barriers", False)
            schedule_graph(graph, self.constraints, use_scheduling_barriers)

        # Align sizes to WG/Tile sizes
        # This pass changes indexing keys, which can interfere with other passes,
        # so it should be called close to the end of pipeline.
        align_index_sizes(graph, self.constraints)

        # Add shared memory barriers.
        add_shared_memory_barriers(graph)

        # Determine grid shape.
        self.grid_type.dims = [1, 1, 1]
        max_workgroup_dim = 2
        aliases = [x.source for x in self.constraints if isinstance(x, SymbolicAlias)]
        for constraint in self.workgroup_constraints:
            if constraint.dim in aliases:
                continue
            dim = (
                constraint.workgroup_dim
                if constraint.workgroup_dim < max_workgroup_dim
                else max_workgroup_dim
            )
            self.grid_type.dims[dim] *= safe_subs(constraint.count, idxc.subs)
        grid = self.grid_type

        root_graph = graph.get_root_graph()
        kernel_sig = kernel_codegen.KernelSignature()
        kernel_sig.add_from_graph_placeholders(root_graph)
        dynamic_symbols = kwargs.get("dynamic_symbols", [])
        kernel_sig.add_from_dynamic_symbols(dynamic_symbols)
        kernel_sig.add_grid(self.grid_type)
        kernel_sig.determine_input_output_buffers(root_graph)

        mb = builder.ModuleBuilder(context=context, module_op=module_op)
        entrypoint_name = self._name
        exe = dispatch_codegen.StreamExecutable(mb, name=entrypoint_name)
        workgroup_size = self.hardware_constraints[0].threads_per_block
        subgroup_size = self.hardware_constraints[0].threads_per_wave

        # Setup LLVM func compilation configs.
        compile_config = kwargs.get("compile_config", {})
        llvm_func_config = {}
        denorm_fp_math_f32 = compile_config.get("denorm_fp_math_f32", None)
        if denorm_fp_math_f32 != None:
            llvm_func_config["denormal-fp-math-f32"] = denorm_fp_math_f32

        waves_per_eu = compile_config.get("waves_per_eu", None)
        if waves_per_eu != None:
            llvm_func_config["amdgpu-waves-per-eu"] = waves_per_eu

        dispatch_entrypoint = exe.define_entrypoint(
            entrypoint_name,
            kernel_sig,
            grid,
            workgroup_size,
            subgroup_size,
            dynamic_symbols,
            llvm_func_config,
        )

        emitter = WaveEmitter(
            dispatch_entrypoint, graph, self.constraints, dynamic_symbols
        )
        emitter.emit(graph.get_root_graph())
        emitter.finish()

        if kwargs.get("canonicalize", False):
            canonicalize_module(mb.module_op)

        return mb, graph, exe, kernel_sig, entrypoint_name

    def test_execute(self, args, kwargs):
        run = kwargs.get("run", False)
        run_bench = kwargs.get("run_bench", False)
        create_vmfb_file = kwargs.get("create_vmfb_file", None)
        dynamic_symbols_map = kwargs.get("dynamic_symbols_map", {})
        dynamic_symbols = kwargs.get("dynamic_symbols", [])
        config = kwargs.get("run_config", None)
        use_scheduling = kwargs.get("schedule", False)
        use_scheduling_barriers = kwargs.get("use_scheduling_barriers", False)

        # Get cached kernel when available.
        cache_enabled = is_cache_enabled()
        if cache_enabled:
            cache_manager = get_cache_manager()
            # TODO: Move use_scheduling, use_scheduling_barriers, etc. into the config so everything is contained there.
            kernel_hash = cache_manager.get_hash(
                self.constraints,
                self._f,
                IndexingContext.current().subs,
                dynamic_symbols,
                config,
                use_scheduling=use_scheduling,
                use_scheduling_barriers=use_scheduling_barriers,
                run_bench=run_bench,
            )
            cached_kernel = cache_manager.load_kernel(kernel_hash)
            if cached_kernel and (run or run_bench):
                invoke_cached_kernel(
                    cached_kernel,
                    args,
                    config,
                    dynamic_symbols,
                    dynamic_symbols_map,
                    run,
                    run_bench,
                )
                return cached_kernel

        # Recompile from kernel scratch if not found in cache.
        (
            mb,
            graph,
            exe,
            kernel_sig,
            entrypoint_name,
        ) = self._trace_and_get_kernel_signature(args, kwargs)

        if run or run_bench or create_vmfb_file:
            host_codegen.isolated_test_call(
                mb, exe, kernel_sig, entrypoint_name, dynamic_symbols
            )
            
            if inject_custom_kernel:
                with mb.context:
                    # This only works properly if the kernel has exactly the 
                    # same sizes, args and so on as the kernel we invoke wave with.
                    mb.module_op = builtin_d.ModuleOp.parse(custom_kernel)
                    print("Injected custom kernel string")

            asm = mb.module_op.get_asm()

            kernel_inputs = []
            kernel_outputs = []
            for arg, b in zip(args, kernel_sig.kernel_buffer_bindings):
                usage = b.kernel_buffer_type.usage
                if usage == kernel_codegen.KernelBufferUsage.INPUT:
                    kernel_inputs.append(arg)

                if usage == kernel_codegen.KernelBufferUsage.OUTPUT:
                    kernel_outputs.append(arg)

            dynamic_symbols_map = kwargs.get("dynamic_symbols_map", {})
            kernel_dynamic_dims = []
            if dynamic_symbols:
                kernel_dynamic_dims = dynamic_symbols_map.values()

            if not config:
                raise ValueError("no config provided")

            compiled_wave_vmfb = compile_to_vmfb(asm, config, run_bench)
            if create_vmfb_file is not None:
                _write_file(create_vmfb_file, "wb", compiled_wave_vmfb)

            kernel_usages = [
                binding.kernel_buffer_type.usage
                for binding in kernel_sig.kernel_buffer_bindings
            ]

            if cache_enabled:
                cache_manager.store_kernel(
                    compiled_wave_vmfb,
                    kernel_usages,
                    mb.module_op.get_asm(),
                    kernel_hash,
                )

            invoke_vmfb(
                compiled_wave_vmfb,
                "isolated_benchmark",
                config,
                kernel_inputs,
                kernel_outputs,
                kernel_dynamic_dims,
                run,
                run_bench,
                inplace=True,
            )

        return mb

    def aot_execute(self, args, kwargs):
        raise NotImplementedError("AOT execution for wave not implemented yet.")

    def eager_execute(self, args, kwargs):
        raise NotImplementedError("Eager execution for wave not implemented yet.")

    def __repr__(self):
        return f"tk.wave @{self._name}[{self.grid_type}]"

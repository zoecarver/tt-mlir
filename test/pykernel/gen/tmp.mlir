#layout = #ttcore.metal_layout<logical_shape = 128x128, dim_alignments = 64x64, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1>
module {
  ttcore.global @lhs = tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout> [0]
  ttcore.global @rhs = tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout> [1]
  func.func @matmul(%arg0: tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout> {d2m.stream = true}, %arg1: tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout> {d2m.stream = true}, %arg2: tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout> {d2m.stream = false}) -> tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout> {
    %0 = d2m.generic {block_factors = [1, 1, 1, 1, 1, 1], grid = #ttcore.grid<2x2>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%arg0, %arg1 : tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout>, tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout>)
        outs(%arg2 : tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout>)  {
    ^datamovement0(%cb0: !d2m.cb<tensor<2x2x!ttcore.tile<32x32, f32>>>, %cb1: !d2m.cb<tensor<2x2x!ttcore.tile<32x32, f32>>>, %cb2: !d2m.cb<tensor<2x2x!ttcore.tile<32x32, f32>>>, %sem0: !d2m.semaphore, %sem1: !d2m.semaphore, %sem2: !d2m.semaphore, %sem3: !d2m.semaphore):
      %c2 = arith.constant 2 : index
      %c2_0 = arith.constant 2 : index
      %c2_1 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %1 = ttcore.get_global @lhs : tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout>
      %core0 = d2m.core_index(0) : index
      %core1 = d2m.core_index(1) : index
      %c0 = arith.constant 0 : index
      %c1_2 = arith.constant 1 : index
      scf.for %arg3 = %c0 to %c2_1 step %c1_2 {
        %c0_3 = arith.constant 0 : index
        %c1_4 = arith.constant 1 : index
        scf.for %arg4 = %c0_3 to %c1 step %c1_4 {
          %2 = d2m.reserve %cb0 : <tensor<2x2x!ttcore.tile<32x32, f32>>> -> tensor<2x2x!ttcore.tile<32x32, f32>>
          %c0_i64 = arith.constant 0 : i64
          %3 = arith.index_cast %c0_i64 : i64 to index
          %4 = arith.cmpi eq, %core1, %3 : index
          scf.if %4 {
            %5 = arith.muli %core0, %c1 : index
            %6 = arith.addi %5, %arg4 : index
            %tx = d2m.dma %1 [%6, %arg3], %2 : (tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout>, tensor<2x2x!ttcore.tile<32x32, f32>>) -> !d2m.mem_tx
            d2m.dma_wait %tx
            %c1_i64 = arith.constant 1 : i64
            %7 = arith.index_cast %c1_i64 : i64 to index
            %8 = arith.subi %c2, %7 : index
            %c0_i64_5 = arith.constant 0 : i64
            %9 = arith.index_cast %c0_i64_5 : i64 to index
            d2m.semaphore_wait %sem0, %8 reset %9
            %c1_i64_6 = arith.constant 1 : i64
            %c1_i64_7 = arith.constant 1 : i64
            %c1_i64_8 = arith.constant 1 : i64
            %10 = arith.index_cast %c1_i64_8 : i64 to index
            %11 = arith.subi %c2_0, %10 : index
            %12 = arith.index_cast %c1_i64_6 : i64 to index
            %13 = arith.index_cast %c1_i64_7 : i64 to index
            %tx_9 = d2m.dma %2, %2 core[%core0, %12] mcast[%13, %11] : (tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>) -> !d2m.mem_tx
            d2m.dma_wait %tx_9
            %c1_i64_10 = arith.constant 1 : i64
            %c1_i64_11 = arith.constant 1 : i64
            %c1_i64_12 = arith.constant 1 : i64
            %c1_i64_13 = arith.constant 1 : i64
            %14 = arith.index_cast %c1_i64_13 : i64 to index
            %15 = arith.subi %c2_0, %14 : index
            %16 = arith.index_cast %c1_i64_10 : i64 to index
            %17 = arith.index_cast %c1_i64_11 : i64 to index
            %18 = arith.index_cast %c1_i64_12 : i64 to index
            d2m.semaphore_set %sem1, %16, core[%core0, %17] mcast[%18, %15]
          } else {
            %c1_i64 = arith.constant 1 : i64
            %c0_i64_5 = arith.constant 0 : i64
            %5 = arith.index_cast %c1_i64 : i64 to index
            %6 = arith.index_cast %c0_i64_5 : i64 to index
            d2m.semaphore_inc %sem0, %5, core[%core0, %6]
            %c1_i64_6 = arith.constant 1 : i64
            %c0_i64_7 = arith.constant 0 : i64
            %7 = arith.index_cast %c1_i64_6 : i64 to index
            %8 = arith.index_cast %c0_i64_7 : i64 to index
            d2m.semaphore_wait %sem1, %7 reset %8
          }
        }
      }
    }, {
    ^datamovement1(%cb0: !d2m.cb<tensor<2x2x!ttcore.tile<32x32, f32>>>, %cb1: !d2m.cb<tensor<2x2x!ttcore.tile<32x32, f32>>>, %cb2: !d2m.cb<tensor<2x2x!ttcore.tile<32x32, f32>>>, %sem0: !d2m.semaphore, %sem1: !d2m.semaphore, %sem2: !d2m.semaphore, %sem3: !d2m.semaphore):
      %c2 = arith.constant 2 : index
      %c2_0 = arith.constant 2 : index
      %c2_1 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c1_2 = arith.constant 1 : index
      %1 = ttcore.get_global @rhs : tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout>
      %core0 = d2m.core_index(0) : index
      %core1 = d2m.core_index(1) : index
      %c0 = arith.constant 0 : index
      %c1_3 = arith.constant 1 : index
      scf.for %arg3 = %c0 to %c2_1 step %c1_3 {
        %c0_4 = arith.constant 0 : index
        %c1_5 = arith.constant 1 : index
        scf.for %arg4 = %c0_4 to %c1 step %c1_5 {
          %c0_6 = arith.constant 0 : index
          %c1_7 = arith.constant 1 : index
          scf.for %arg5 = %c0_6 to %c1_2 step %c1_7 {
            %2 = d2m.reserve %cb1 : <tensor<2x2x!ttcore.tile<32x32, f32>>> -> tensor<2x2x!ttcore.tile<32x32, f32>>
            %c0_i64 = arith.constant 0 : i64
            %3 = arith.index_cast %c0_i64 : i64 to index
            %4 = arith.cmpi eq, %core0, %3 : index
            scf.if %4 {
              %5 = arith.muli %core1, %c1_2 : index
              %6 = arith.addi %5, %arg5 : index
              %tx = d2m.dma %1 [%arg3, %6], %2 : (tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout>, tensor<2x2x!ttcore.tile<32x32, f32>>) -> !d2m.mem_tx
              d2m.dma_wait %tx
              %c1_i64 = arith.constant 1 : i64
              %7 = arith.index_cast %c1_i64 : i64 to index
              %8 = arith.subi %c2, %7 : index
              %c0_i64_8 = arith.constant 0 : i64
              %9 = arith.index_cast %c0_i64_8 : i64 to index
              d2m.semaphore_wait %sem2, %8 reset %9
              %c1_i64_9 = arith.constant 1 : i64
              %c1_i64_10 = arith.constant 1 : i64
              %10 = arith.index_cast %c1_i64_10 : i64 to index
              %11 = arith.subi %c2_0, %10 : index
              %c1_i64_11 = arith.constant 1 : i64
              %12 = arith.index_cast %c1_i64_9 : i64 to index
              %13 = arith.index_cast %c1_i64_11 : i64 to index
              %tx_12 = d2m.dma %2, %2 core[%12, %core1] mcast[%11, %13] : (tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>) -> !d2m.mem_tx
              d2m.dma_wait %tx_12
              %c1_i64_13 = arith.constant 1 : i64
              %c1_i64_14 = arith.constant 1 : i64
              %c1_i64_15 = arith.constant 1 : i64
              %14 = arith.index_cast %c1_i64_15 : i64 to index
              %15 = arith.subi %c2_0, %14 : index
              %c1_i64_16 = arith.constant 1 : i64
              %16 = arith.index_cast %c1_i64_13 : i64 to index
              %17 = arith.index_cast %c1_i64_14 : i64 to index
              %18 = arith.index_cast %c1_i64_16 : i64 to index
              d2m.semaphore_set %sem3, %16, core[%17, %core1] mcast[%15, %18]
            } else {
              %c1_i64 = arith.constant 1 : i64
              %c0_i64_8 = arith.constant 0 : i64
              %5 = arith.index_cast %c1_i64 : i64 to index
              %6 = arith.index_cast %c0_i64_8 : i64 to index
              d2m.semaphore_inc %sem2, %5, core[%6, %core1]
              %c1_i64_9 = arith.constant 1 : i64
              %c0_i64_10 = arith.constant 0 : i64
              %7 = arith.index_cast %c1_i64_9 : i64 to index
              %8 = arith.index_cast %c0_i64_10 : i64 to index
              d2m.semaphore_wait %sem3, %7 reset %8
            }
          }
        }
      }
    }, {
    ^compute0(%cb0: !d2m.cb<tensor<2x2x!ttcore.tile<32x32, f32>>>, %cb1: !d2m.cb<tensor<2x2x!ttcore.tile<32x32, f32>>>, %cb2: !d2m.cb<tensor<2x2x!ttcore.tile<32x32, f32>>>, %sem0: !d2m.semaphore, %sem1: !d2m.semaphore, %sem2: !d2m.semaphore, %sem3: !d2m.semaphore):
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c1_0 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c1_1 = arith.constant 1 : index
      scf.for %arg3 = %c0 to %c2 step %c1_1 {
        %c0_2 = arith.constant 0 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg4 = %c0_2 to %c1 step %c1_3 {
          %1 = d2m.pop %cb0 : <tensor<2x2x!ttcore.tile<32x32, f32>>> -> tensor<2x2x!ttcore.tile<32x32, f32>>
          %c0_4 = arith.constant 0 : index
          %c1_5 = arith.constant 1 : index
          scf.for %arg5 = %c0_4 to %c1_0 step %c1_5 {
            %2 = d2m.pop %cb1 : <tensor<2x2x!ttcore.tile<32x32, f32>>> -> tensor<2x2x!ttcore.tile<32x32, f32>>
            %3 = d2m.reserve %cb2 : <tensor<2x2x!ttcore.tile<32x32, f32>>> -> tensor<2x2x!ttcore.tile<32x32, f32>>
            %4 = d2m.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
            "d2m.tile_matmul_block"(%1, %2, %4) : (tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>) -> ()
            d2m.store %3, %4 : tensor<2x2x!ttcore.tile<32x32, f32>>
            %5 = d2m.pop %cb2 : <tensor<2x2x!ttcore.tile<32x32, f32>>> -> tensor<2x2x!ttcore.tile<32x32, f32>>
          }
        }
      }
    } : tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout>
    return %0 : tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout>
  }
}


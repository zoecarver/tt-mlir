// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/Passes.h"
#include "ttmlir/Dialect/D2M/Utils/Utils.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MBUFFERIZEFUNCTIONARGS
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

class D2MBufferizeFunctionArgs
    : public impl::D2MBufferizeFunctionArgsBase<D2MBufferizeFunctionArgs> {
public:
  using impl::D2MBufferizeFunctionArgsBase<
      D2MBufferizeFunctionArgs>::D2MBufferizeFunctionArgsBase;

  void runOnOperation() final {
    ModuleOp module = getOperation();

    // Walk through all functions
    module.walk([&](func::FuncOp funcOp) {
      // Skip if function has no body
      if (funcOp.empty()) {
        return;
      }

      OpBuilder builder(funcOp.getContext());
      SmallVector<Type> newArgTypes;
      SmallVector<Type> newResultTypes;
      bool needsUpdate = false;

      // Process each function argument
      // After bufferization, arguments are bare memrefs without layout/memory space attrs
      // We need to add them back based on stream_layout ops in the function body
      for (auto [idx, arg] : llvm::enumerate(funcOp.getArguments())) {
        Type argType = arg.getType();
        auto memrefType = dyn_cast<MemRefType>(argType);

        // Skip if not a memref
        if (!memrefType) {
          newArgTypes.push_back(argType);
          continue;
        }

        // Check if this memref already has attributes
        if (memrefType.getLayout() && !isa<AffineMapAttr>(memrefType.getLayout())) {
          // Already has non-identity layout, keep it
          newArgTypes.push_back(argType);
          continue;
        }

        if (memrefType.getMemorySpace()) {
          // Already has memory space, keep it
          newArgTypes.push_back(argType);
          continue;
        }

        // Look for stream_layout ops that use this argument
        // to infer the correct type
        MemRefType newMemrefType = memrefType;
        bool foundStreamLayout = false;
        for (auto user : arg.getUsers()) {
          if (auto streamLayout = dyn_cast<d2m::StreamLayoutOp>(user)) {
            // Get the storage operand type (has ShardLayoutAttr)
            auto storageType = streamLayout.getStorage().getType();
            if (auto storageMemrefType = dyn_cast<MemRefType>(storageType)) {
              // Use the storage's layout and memory space for the function arg
              newMemrefType = MemRefType::get(
                  memrefType.getShape(),
                  memrefType.getElementType(),
                  storageMemrefType.getLayout(),
                  storageMemrefType.getMemorySpace());
              needsUpdate = true;
              foundStreamLayout = true;
              break;
            }
          }
        }

        // If we didn't find a stream_layout op, add default memory space (L1)
        if (!foundStreamLayout) {
          auto ctx = builder.getContext();
          auto memSpace = ttcore::MemorySpaceAttr::get(ctx, ttcore::MemorySpace::DeviceL1);
          newMemrefType = MemRefType::get(
              memrefType.getShape(),
              memrefType.getElementType(),
              memrefType.getLayout(),
              memSpace);
          needsUpdate = true;
        }

        newArgTypes.push_back(newMemrefType);
      }

      // Process function result types
      for (auto resultType : funcOp.getResultTypes()) {
        auto memrefType = dyn_cast<MemRefType>(resultType);

        // Skip if not a memref
        if (!memrefType) {
          newResultTypes.push_back(resultType);
          continue;
        }

        // Check if this memref already has attributes
        if (memrefType.getLayout() && !isa<AffineMapAttr>(memrefType.getLayout())) {
          // Already has non-identity layout, keep it
          newResultTypes.push_back(resultType);
          continue;
        }

        if (memrefType.getMemorySpace()) {
          // Already has memory space, keep it
          newResultTypes.push_back(resultType);
          continue;
        }

        // Add default memory space (L1)
        auto ctx = builder.getContext();
        auto memSpace = ttcore::MemorySpaceAttr::get(ctx, ttcore::MemorySpace::DeviceL1);
        auto newMemrefType = MemRefType::get(
            memrefType.getShape(),
            memrefType.getElementType(),
            memrefType.getLayout(),
            memSpace);
        newResultTypes.push_back(newMemrefType);
        needsUpdate = true;
      }

      // If no changes needed, skip this function
      if (!needsUpdate) {
        return;
      }

      // Create new function type
      auto newFuncType = builder.getFunctionType(newArgTypes, newResultTypes);

      // Update function signature
      funcOp.setFunctionType(newFuncType);

      // Update the entry block argument types
      Block &entryBlock = funcOp.front();
      for (auto [idx, arg] : llvm::enumerate(entryBlock.getArguments())) {
        arg.setType(newArgTypes[idx]);
      }
    });
  }
};

} // namespace

} // namespace mlir::tt::d2m

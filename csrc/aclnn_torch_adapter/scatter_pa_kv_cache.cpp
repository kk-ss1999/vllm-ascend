// Copyright (c) 2023 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Temporary port of ordinary EXEC_NPU_CMD from op-plugin b80a2fdf.
// Only the VA scatter binding uses this dispatcher. Keep PTA's exported tensor
// conversion/queue APIs, and isolate its unexported cache state in this TU.
#include <array>
#include <cstring>
#include <string>
#include <third_party/op-plugin/op_plugin/utils/op_api_common_base.h>
#include "torch_npu/csrc/core/npu/NPUFunctions.h"
#include "torch_npu/csrc/framework/OpParamMaker.h"

namespace vllm_ascend {
namespace scatter_dispatch {
constexpr int g_hash_buf_size = 8192;
constexpr int g_hash_buf_max_size = g_hash_buf_size + 1024;
thread_local char g_hash_buf[g_hash_buf_size];
thread_local int g_hash_offset = 0;
constexpr int g_rShift33Bits = 33;
constexpr uint64_t MIX_STEP1 = 18397679294719823053LLU;
constexpr uint64_t MIX_STEP2 = 14181476777654086739LLU;
using AddTensorAddrToCachedList = void (*)(void*);

// The snapshot functions/TLS are private in PTA post7. Capture the same level
// and boolean, then apply it through PTA's exported API (which keeps PTA's
// backend selection and shared deterministic-launch mutex).
struct ScatterDeterministicSnapshot {
    bool deterministic_algorithms_enabled;
    uint32_t requested_level;
    uint32_t effective_level;
};
ScatterDeterministicSnapshot CaptureScatterDeterministicSnapshot() {
    ScatterDeterministicSnapshot snapshot;
    snapshot.deterministic_algorithms_enabled = at::globalContext().deterministicAlgorithms();
    snapshot.requested_level = c10_npu::GetDeterministicLevel();
    snapshot.effective_level = 0;
    if (snapshot.deterministic_algorithms_enabled) {
        snapshot.effective_level = snapshot.requested_level == 0 ? 1 : snapshot.requested_level;
    }
    return snapshot;
}
void ApplyScatterDeterministicSnapshot(const ScatterDeterministicSnapshot& snapshot, bool is_opapi) {
    // PTA's snapshot TLS is private. Always apply through the exported API:
    // a separate zero-level shortcut could become stale after other PTA ops.
    at_npu::native::ApplyDeterministicLevel(
        snapshot.requested_level, snapshot.deterministic_algorithms_enabled, is_opapi);
}
inline void CheckNpuTensorValid(const at::Tensor& at_tensor) {
  TORCH_CHECK(
      torch_npu::utils::is_npu(at_tensor),
      "Expected all tensors to be on the same device. "
      "Expected NPU tensor, please check whether the input tensor device is correct.",
      OPS_ERROR(ErrCode::TYPE));
  TORCH_CHECK(
      at_tensor.numel() <= 0 || at_tensor.unsafeGetTensorImpl()->storage_initialized(),
      "The tensor has a non-zero number of elements, but its data is not allocated yet.",
      OPS_ERROR(ErrCode::VALUE));
}
void ScatterMemcpyToBufImpl(const void* data, size_t size) {
  if (g_hash_offset + size > g_hash_buf_size) {
    g_hash_offset = g_hash_buf_max_size;
    return;
  }
  memcpy(g_hash_buf + g_hash_offset, data, size);
  g_hash_offset += size;
}

#define SCATTER_MEMCPY_TO_BUF(data, size) ScatterMemcpyToBufImpl(data, size)
void scatter_add_param_to_buf(const at::Tensor& at_tensor) {
  static const auto addTensorAddrToCachedListAddr = GetOpApiFuncAddr("AddTensorAddrToCachedList");
  TORCH_CHECK(addTensorAddrToCachedListAddr != nullptr, "GetOpApiFuncAddr failed.", OPS_ERROR(ErrCode::PTR));
  AddTensorAddrToCachedList addTensorAddrToCachedListFunc =
      reinterpret_cast<AddTensorAddrToCachedList>(addTensorAddrToCachedListAddr);
  if (!at_tensor.defined()) {
    SCATTER_MEMCPY_TO_BUF(",", 1);
    return;
  }
  CheckNpuTensorValid(at_tensor);
  if (at_npu::native::OpPreparation::is_scalar_wrapped_to_tensor(at_tensor)) {
    g_hash_offset = g_hash_buf_max_size;
    return;
  }
  // view shape
  SCATTER_MEMCPY_TO_BUF(at_tensor.sizes().data(), static_cast<int64_t>(at_tensor.sizes().size() * sizeof(int64_t)));
  // data type
  auto st = at_tensor.scalar_type();
  SCATTER_MEMCPY_TO_BUF(&st, sizeof(st));
  // seperator
  SCATTER_MEMCPY_TO_BUF(",", 1);
  // strides
  SCATTER_MEMCPY_TO_BUF(at_tensor.strides().data(), static_cast<int64_t>(at_tensor.sizes().size() * sizeof(int64_t)));
  // offset
  auto so = at_tensor.storage_offset();
  SCATTER_MEMCPY_TO_BUF(&so, sizeof(so));
  // storage shape
  aclDataType acl_data_type = at_npu::native::OpPreparation::convert_to_acl_data_type(st);
  c10::SmallVector<int64_t, 5> storageDims;
  if (acl_data_type != ACL_STRING) {
    TORCH_CHECK(at_tensor.itemsize() > 0, "the itemsize of tensor must be greater than 0.", OPS_ERROR(ErrCode::PARAM));
    storageDims.push_back(at_tensor.storage().nbytes() / at_tensor.itemsize());
  }
  SCATTER_MEMCPY_TO_BUF(storageDims.data(), static_cast<int64_t>(storageDims.size() * sizeof(int64_t)));
  aclFormat npu_format_ = ACL_FORMAT_ND;
  auto* storage_impl = at_tensor.storage().unsafeGetStorageImpl();
  if (typeid(*storage_impl) == typeid(torch_npu::NPUStorageImpl)) {
    npu_format_ = torch_npu::NPUBridge::GetNpuStorageImpl(at_tensor)->npu_desc_.npu_format_;
  }
  SCATTER_MEMCPY_TO_BUF(&npu_format_, sizeof(npu_format_));

  addTensorAddrToCachedListFunc(const_cast<void*>(at_tensor.storage().data()));
}

void scatter_add_param_to_buf(const at::IntArrayRef& at_array) {
  SCATTER_MEMCPY_TO_BUF(at_array.data(), static_cast<int64_t>(at_array.size() * sizeof(int64_t)));
  auto counter = at_array.size();
  SCATTER_MEMCPY_TO_BUF(&counter, sizeof(counter));
}

void scatter_add_param_to_buf(const c10::optional<at::Tensor>& opt_tensor) {
  if (opt_tensor.has_value() && opt_tensor.value().defined()) {
    scatter_add_param_to_buf(opt_tensor.value());
  } else {
    SCATTER_MEMCPY_TO_BUF(",", 1);
  }
}

void scatter_add_param_to_buf(const string& s) {
  SCATTER_MEMCPY_TO_BUF(s.c_str(), static_cast<int64_t>(s.size()));
}

void scatter_add_param_to_buf(char* c) {
  SCATTER_MEMCPY_TO_BUF(c, strlen(c));
  auto counter = strlen(c);
  SCATTER_MEMCPY_TO_BUF(&counter, sizeof(counter));
}

void scatter_add_param_to_buf_v2(TensorStructPtr at_tensor) {
  static const auto addTensorAddrToCachedListAddr = GetOpApiFuncAddr("AddTensorAddrToCachedList");
  TORCH_CHECK(addTensorAddrToCachedListAddr != nullptr, "GetOpApiFuncAddr failed.", OPS_ERROR(ErrCode::PTR));
  AddTensorAddrToCachedList addTensorAddrToCachedListFunc =
      reinterpret_cast<AddTensorAddrToCachedList>(addTensorAddrToCachedListAddr);
  if (at_tensor == nullptr) {
    SCATTER_MEMCPY_TO_BUF(",", 1);
    return;
  }
  // view shape
  SCATTER_MEMCPY_TO_BUF((*at_tensor).sizes.data(), static_cast<int64_t>((*at_tensor).sizes.size() * sizeof(int64_t)));
  // data type
  auto acl_data_type = (*at_tensor).acl_type;
  SCATTER_MEMCPY_TO_BUF(&acl_data_type, sizeof(acl_data_type));
  // seperator
  SCATTER_MEMCPY_TO_BUF(",", 1);
  // strides
  SCATTER_MEMCPY_TO_BUF((*at_tensor).strides.data(), static_cast<int64_t>((*at_tensor).sizes.size() * sizeof(int64_t)));
  // offset
  auto so = (*at_tensor).storage_offset;
  SCATTER_MEMCPY_TO_BUF(&so, sizeof(so));
  // storage shape
  c10::SmallVector<int64_t, 5> storageDims;
  if (acl_data_type != ACL_STRING) {
    TORCH_CHECK((*at_tensor).itemsize > 0, "the itemsize of tensor must be greater than 0.", OPS_ERROR(ErrCode::PARAM));
    storageDims.push_back((*at_tensor).nbytes / (*at_tensor).itemsize);
  }
  SCATTER_MEMCPY_TO_BUF(storageDims.data(), static_cast<int64_t>(storageDims.size() * sizeof(int64_t)));
  auto npu_format_v2 = (*at_tensor).acl_format;
  SCATTER_MEMCPY_TO_BUF(&npu_format_v2, sizeof(npu_format_v2));

  addTensorAddrToCachedListFunc((*at_tensor).data_ptr);
}

void scatter_add_param_to_buf_v2(const std::vector<int64_t>& at_array) {
  SCATTER_MEMCPY_TO_BUF(at_array.data(), static_cast<int64_t>(at_array.size() * sizeof(int64_t)));
  auto counter = at_array.size();
  SCATTER_MEMCPY_TO_BUF(&counter, sizeof(counter));
}

void scatter_add_param_to_buf_v2(const string& s) {
  SCATTER_MEMCPY_TO_BUF(s.c_str(), static_cast<int64_t>(s.size()));
}
template <typename T> void scatter_add_param_to_buf(const T& value) {
    SCATTER_MEMCPY_TO_BUF(&value, sizeof(T));
}
inline void scatter_add_param_to_buf() {}
template <typename T, typename... Args>
void scatter_add_param_to_buf(const T& arg, Args&... args) {
    scatter_add_param_to_buf(arg);
    scatter_add_param_to_buf(args...);
}
template <typename T> void scatter_add_param_to_buf_v2(const T& value) {
    SCATTER_MEMCPY_TO_BUF(&value, sizeof(T));
}
template <typename... Ts, size_t... i>
void scatter_add_params_to_buf_v2(const std::tuple<Ts...>& t, std::index_sequence<i...>) {
    (scatter_add_param_to_buf_v2(std::get<i>(t)), ...);
}
inline uint64_t rotating_left(uint64_t x, uint8_t n) {
  return (x << n) | (x >> (64 - n));
}

inline uint64_t mixture(uint64_t x) {
  // constants step1(18397679294719823053) and step2(14181476777654086739) are used to allow
  // hash values to be more evenly distributed after multiplication.
  x ^= x >> g_rShift33Bits;
  x *= MIX_STEP1;
  x ^= x >> g_rShift33Bits;
  x *= MIX_STEP2;
  x ^= x >> g_rShift33Bits;

  return x;
}

// MurmurHash3 was written by Austin Appleby, and is placed in the public
// domain. The author hereby disclaims copyright to this source code.
uint64_t gen_hash(const void* key, const int len, const uint32_t seed = 0xdeadb0d7) {
  const uint8_t* data = (const uint8_t*)key;
  // the length of each block is 16 bytes
  const int block_num = len / 16;
  // has and hax are literal appromix to hash, and hax is the return value of this function.
  uint64_t has = seed;
  uint64_t hax = seed;

  // use 9782798678568883157 and 5545529020109919103 for blocking and obfuscation of input data
  const uint64_t c1 = 9782798678568883157LLU;
  const uint64_t c2 = 5545529020109919103LLU;

  const uint64_t* blocks = (const uint64_t*)(data);

  for (int i = 0; i < block_num; i++) {
    int even_num = 2;
    uint64_t tmp1 = blocks[i * even_num];
    uint64_t tmp2 = blocks[i * even_num + 1];

    int8_t bits_31 = 31;
    tmp1 *= c1;
    tmp1 = rotating_left(tmp1, bits_31);
    tmp1 *= c2;
    has ^= tmp1;

    int8_t bits_27 = 27;
    has = rotating_left(has, bits_27);
    has += hax;
    // increase randomness by mul by 5 and adding a constant
    has = has * 5 + 1390208809;

    int8_t bits_33 = 33;
    tmp2 *= c2;
    tmp2 = rotating_left(tmp2, bits_33);
    tmp2 *= c1;
    hax ^= tmp2;

    hax = rotating_left(hax, bits_31);
    hax += has;
    // increase randomness by mul by 5 and adding a constant
    hax = hax * 5 + 944331445;
  }

  // the length of each block is 16 bytes
  const uint8_t* tail = (const uint8_t*)(data + block_num * 16);
  uint64_t t1 = 0;
  uint64_t t2 = 0;
  // because the size of a block is 16, different offsets are calculated for tail blocks
  // for different sizes
  switch (static_cast<uint64_t>(len) & 15) {
    case 15:
      t2 ^= ((uint64_t)tail[14]) << 48;
      [[fallthrough]];
      ;
    case 14:
      t2 ^= ((uint64_t)tail[13]) << 40;
      [[fallthrough]];
      ;
    case 13:
      t2 ^= ((uint64_t)tail[12]) << 32;
      [[fallthrough]];
      ;
    case 12:
      t2 ^= ((uint64_t)tail[11]) << 24;
      [[fallthrough]];
      ;
    case 11:
      t2 ^= ((uint64_t)tail[10]) << 16;
      [[fallthrough]];
      ;
    case 10:
      t2 ^= ((uint64_t)tail[9]) << 8;
      [[fallthrough]];
      ;
    case 9:
      t2 ^= ((uint64_t)tail[8]) << 0;
      t2 *= c2;
      t2 = rotating_left(t2, 33);
      t2 *= c1;
      hax ^= t2;
      [[fallthrough]];
      ;
    case 8:
      t1 ^= ((uint64_t)tail[7]) << 56;
      [[fallthrough]];
      ;
    case 7:
      t1 ^= ((uint64_t)tail[6]) << 48;
      [[fallthrough]];
      ;
    case 6:
      t1 ^= ((uint64_t)tail[5]) << 40;
      [[fallthrough]];
      ;
    case 5:
      t1 ^= ((uint64_t)tail[4]) << 32;
      [[fallthrough]];
      ;
    case 4:
      t1 ^= ((uint64_t)tail[3]) << 24;
      [[fallthrough]];
      ;
    case 3:
      t1 ^= ((uint64_t)tail[2]) << 16;
      [[fallthrough]];
      ;
    case 2:
      t1 ^= ((uint64_t)tail[1]) << 8;
      [[fallthrough]];
      ;
    case 1:
      t1 ^= ((uint64_t)tail[0]) << 0;
      t1 *= c1;
      t1 = rotating_left(t1, 31);
      t1 *= c2;
      has ^= t1;
      [[fallthrough]];
      ;
    default:
      break;
  };

  has ^= static_cast<uint64_t>(len);
  hax ^= static_cast<uint64_t>(len);

  has += hax;
  hax += has;

  has = mixture(has);
  hax = mixture(hax);

  has += hax;
  hax += has;
  return hax;
}

uint64_t calc_hash_id() {
  if (g_hash_offset == g_hash_buf_max_size) {
    return 0;
  }
  uint64_t hash_id = gen_hash(g_hash_buf, g_hash_offset);
  return hash_id;
}

typedef int (*InitHugeMemThreadLocal)(void*, bool);
typedef void (*UnInitHugeMemThreadLocal)(void*, bool);
typedef void (*ReleaseHugeMem)(void*, bool);
typedef aclOpExecutor* (*PTAGetExecCache)(uint64_t, uint64_t*);
typedef aclOpExecutor* (*PTAFindExecCache)(uint8_t*, size_t, uint64_t*);
typedef void (*InitPTACacheThreadLocal)();
typedef void (*SetPTAHashKey)(uint64_t);
typedef void (*SetPTACacheHashKey)(uint8_t*, size_t);
typedef bool (*CanUsePTACache)(const char*);
typedef void (*UnInitPTACacheThreadLocal)();

inline void UnInitCacheThreadLocal() {
  static const auto unInitPTACacheThreadLocalAddr = GetOpApiFuncAddr("UnInitPTACacheThreadLocal");
  UnInitPTACacheThreadLocal unInitPTACacheThreadLocalFunc =
      reinterpret_cast<UnInitPTACacheThreadLocal>(unInitPTACacheThreadLocalAddr);
  if (unInitPTACacheThreadLocalFunc) {
    unInitPTACacheThreadLocalFunc();
  }
}

template <typename... Args>
bool hit_cache(
    aclrtStream acl_stream,
    const char* aclnn_api,
    void* phrase2,
    const ScatterDeterministicSnapshot& snapshot,
    Args&&... args) {
  static const auto ptaGetExecCacheAddr = GetOpApiFuncAddr("PTAGetExecCache");
  static const auto initPTACacheThreadLocalAddr = GetOpApiFuncAddr("InitPTACacheThreadLocal");
  static const auto setPTAHashKeyAddr = GetOpApiFuncAddr("SetPTAHashKey");
  static const auto canUsePTACacheAddr = GetOpApiFuncAddr("CanUsePTACache");
  PTAGetExecCache ptaGetExecCacheFunc = reinterpret_cast<PTAGetExecCache>(ptaGetExecCacheAddr);
  InitPTACacheThreadLocal initPTACacheThreadLocalFunc =
      reinterpret_cast<InitPTACacheThreadLocal>(initPTACacheThreadLocalAddr);
  SetPTAHashKey setPTAHashKeyFunc = reinterpret_cast<SetPTAHashKey>(setPTAHashKeyAddr);
  CanUsePTACache canUsePTACacheFunc = reinterpret_cast<CanUsePTACache>(canUsePTACacheAddr);
  bool has_func = ptaGetExecCacheFunc && initPTACacheThreadLocalFunc && setPTAHashKeyFunc;
  bool can_use = canUsePTACacheFunc && canUsePTACacheFunc(aclnn_api);
  ApplyScatterDeterministicSnapshot(snapshot, true);
  if (!has_func || !can_use) {
    return false;
  }
  uint64_t workspace_size = 0;
  uint64_t* workspace_size_addr = &workspace_size;
  initPTACacheThreadLocalFunc();
  g_hash_offset = 0;
  auto deterministic_level = snapshot.effective_level;
  if (c10_npu::is_core_control_enabled()) {
    auto aic_num = c10_npu::GetResInCurrentThread(c10_npu::acl::ACL_RT_DEV_RES_CUBE_CORE);
    auto aiv_num = c10_npu::GetResInCurrentThread(c10_npu::acl::ACL_RT_DEV_RES_VECTOR_CORE);
    scatter_add_param_to_buf(aic_num);
    scatter_add_param_to_buf(aiv_num);
  }
  auto device = c10_npu::current_device();
  scatter_add_param_to_buf(deterministic_level);
  scatter_add_param_to_buf(std::string(aclnn_api), args...);
  scatter_add_param_to_buf(device);
  scatter_add_param_to_buf((uintptr_t)acl_stream);
  uint64_t hashId = calc_hash_id();
  setPTAHashKeyFunc(hashId);
  aclOpExecutor* executor = ptaGetExecCacheFunc(hashId, workspace_size_addr);
  if (executor == nullptr) {
    return false;
  }
  void* workspace_addr = nullptr;
  at::Tensor workspace_tensor;
  if (workspace_size != 0) {
    workspace_tensor = at_npu::native::OpPreparation::unsafe_empty_workspace(workspace_size);
    workspace_addr = const_cast<void*>(workspace_tensor.storage().data());
  }
  auto acl_call = [workspace_addr, workspace_size, acl_stream, executor, phrase2]() -> int {
    OpApiFunc opApiFunc = reinterpret_cast<OpApiFunc>(phrase2);
    auto api_ret = opApiFunc(workspace_addr, workspace_size, executor, acl_stream);
    NPU_CHECK_ERROR(api_ret, "call failed");
    return api_ret;
  };
  at_npu::native::OpCommand::RunOpApiV2(aclnn_api, acl_call);
  UnInitCacheThreadLocal();
  return true;
}

template <typename... Ts>
bool hit_cache_v2(
    aclrtStream acl_stream,
    const char* aclnn_api,
    void* phrase2,
    const std::tuple<Ts...>& args,
    int* api_ret,
    uint32_t aic_num,
    uint32_t aiv_num,
    const ScatterDeterministicSnapshot& snapshot) {
  static const auto ptaFindExecCacheAddr = GetOpApiFuncAddr("PTAFindExecCache");
  static const auto initPTACacheThreadLocalAddr = GetOpApiFuncAddr("InitPTACacheThreadLocal");
  static const auto setPTACacheHashKeyAddr = GetOpApiFuncAddr("SetPTACacheHashKey");
  static const auto canUsePTACacheAddr = GetOpApiFuncAddr("CanUsePTACache");
  PTAFindExecCache ptaFindExecCacheFunc = reinterpret_cast<PTAFindExecCache>(ptaFindExecCacheAddr);
  InitPTACacheThreadLocal initPTACacheThreadLocalFunc =
      reinterpret_cast<InitPTACacheThreadLocal>(initPTACacheThreadLocalAddr);
  SetPTACacheHashKey setPTACacheHashKeyFunc = reinterpret_cast<SetPTACacheHashKey>(setPTACacheHashKeyAddr);
  CanUsePTACache canUsePTACacheFunc = reinterpret_cast<CanUsePTACache>(canUsePTACacheAddr);
  bool has_func = ptaFindExecCacheFunc && initPTACacheThreadLocalFunc && setPTACacheHashKeyFunc;
  bool can_use = canUsePTACacheFunc && canUsePTACacheFunc(aclnn_api);
  ApplyScatterDeterministicSnapshot(snapshot, true);
  if (!has_func || !can_use) {
    return false;
  }
  uint64_t workspace_size = 0;
  uint64_t* workspace_size_addr = &workspace_size;
  initPTACacheThreadLocalFunc();
  g_hash_offset = 0;
  uint32_t deterministic_level = snapshot.effective_level;
  scatter_add_param_to_buf_v2(deterministic_level);
  if (aic_num != UINT32_MAX && aiv_num != UINT32_MAX) {
    scatter_add_param_to_buf_v2(aic_num);
    scatter_add_param_to_buf_v2(aiv_num);
  }
  scatter_add_param_to_buf_v2(std::string(aclnn_api));
  scatter_add_params_to_buf_v2(args, std::make_index_sequence<sizeof...(Ts)>{});
  scatter_add_param_to_buf_v2((uintptr_t)acl_stream);
  if (g_hash_offset == g_hash_buf_max_size) {
    setPTACacheHashKeyFunc(nullptr, 0);
  } else {
    setPTACacheHashKeyFunc(reinterpret_cast<uint8_t*>(g_hash_buf), g_hash_offset);
  }
  aclOpExecutor* executor =
      ptaFindExecCacheFunc(reinterpret_cast<uint8_t*>(g_hash_buf), g_hash_offset, workspace_size_addr);
  if (executor == nullptr) {
    return false;
  }
  void* workspace_addr = nullptr;
  at::Tensor workspace_tensor;
  if (workspace_size != 0) {
    workspace_tensor = at_npu::native::OpPreparation::unsafe_empty_workspace(workspace_size, acl_stream);
    workspace_addr = const_cast<void*>(workspace_tensor.storage().data());
  }
  OpApiFunc opApiFunc = reinterpret_cast<OpApiFunc>(phrase2);
  *api_ret = opApiFunc(workspace_addr, workspace_size, executor, acl_stream);
  NPU_CHECK_ERROR(*api_ret, "call failed");
  UnInitCacheThreadLocal();
  return true;
}

#define EXEC_NPU_CMD_SCATTER_V1(aclnn_api, ...)                                                                 \
  do {                                                                                                  \
    static const auto getWorkspaceSizeFuncAddr = GetOpApiFuncAddr(#aclnn_api "GetWorkspaceSize");       \
    static const auto opApiFuncAddr = GetOpApiFuncAddr(#aclnn_api);                                     \
    static const auto initMemAddr = GetOpApiFuncAddr("InitHugeMemThreadLocal");                         \
    static const auto unInitMemAddr = GetOpApiFuncAddr("UnInitHugeMemThreadLocal");                     \
    static const auto releaseMemAddr = GetOpApiFuncAddr("ReleaseHugeMem");                              \
    TORCH_CHECK(                                                                                        \
        getWorkspaceSizeFuncAddr != nullptr && opApiFuncAddr != nullptr,                                \
        #aclnn_api,                                                                                     \
        " or ",                                                                                         \
        #aclnn_api "GetWorkspaceSize",                                                                  \
        " not in ",                                                                                     \
        GetOpApiLibName(),                                                                              \
        ", or ",                                                                                        \
        GetOpApiLibName(),                                                                              \
        " not found.",                                                                                  \
        OPS_ERROR(ErrCode::PTR));                                                                       \
    auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);                                     \
    if (c10_npu::check_enqueue_need_use(acl_stream)) {                                                  \
      c10_npu::UseStreamResInCurrentThread(acl_stream);                                                 \
    }                                                                                                   \
    OP_EXEC_LOG_WITH_TASK_QUEUE(#aclnn_api, "EXEC_NPU_CMD", "1", acl_stream, __VA_ARGS__);              \
    uint64_t workspace_size = 0;                                                                        \
    uint64_t* workspace_size_addr = &workspace_size;                                                    \
    aclOpExecutor* executor = nullptr;                                                                  \
    aclOpExecutor** executor_addr = &executor;                                                          \
    InitHugeMemThreadLocal initMemFunc = reinterpret_cast<InitHugeMemThreadLocal>(initMemAddr);         \
    UnInitHugeMemThreadLocal unInitMemFunc = reinterpret_cast<UnInitHugeMemThreadLocal>(unInitMemAddr); \
    auto snapshot = CaptureScatterDeterministicSnapshot();                                            \
    if (hit_cache(acl_stream, #aclnn_api, opApiFuncAddr, snapshot, __VA_ARGS__)) {                      \
      break;                                                                                            \
    }                                                                                                   \
    if (initMemFunc) {                                                                                  \
      initMemFunc(nullptr, false);                                                                      \
    }                                                                                                   \
    auto converted_params = ConvertTypes(__VA_ARGS__, workspace_size_addr, executor_addr);              \
    static auto getWorkspaceSizeFunc = ConvertToOpApiFunc(converted_params, getWorkspaceSizeFuncAddr);  \
    auto workspace_status = call(getWorkspaceSizeFunc, converted_params);                               \
    NPU_CHECK_ERROR(workspace_status, "call " #aclnn_api " failed");                                    \
    void* workspace_addr = nullptr;                                                                     \
    at::Tensor workspace_tensor;                                                                        \
    if (workspace_size != 0) {                                                                          \
      workspace_tensor = at_npu::native::OpPreparation::unsafe_empty_workspace(workspace_size);         \
      workspace_addr = const_cast<void*>(workspace_tensor.storage().data());                            \
    }                                                                                                   \
    auto acl_call = [converted_params, workspace_addr, workspace_size, acl_stream, executor]() -> int { \
      if (c10_npu::check_dequeue_need_use(acl_stream)) {                                                \
        c10_npu::UseStreamResInCurrentThread(acl_stream);                                               \
      }                                                                                                 \
      OpApiFunc opApiFunc = reinterpret_cast<OpApiFunc>(opApiFuncAddr);                                 \
      auto api_ret = opApiFunc(workspace_addr, workspace_size, executor, acl_stream);                   \
      NPU_CHECK_ERROR(api_ret, "call " #aclnn_api " failed");                                           \
      ReleaseConvertTypes(converted_params);                                                            \
      ReleaseHugeMem releaseMemFunc = reinterpret_cast<ReleaseHugeMem>(releaseMemAddr);                 \
      if (releaseMemFunc) {                                                                             \
        releaseMemFunc(nullptr, false);                                                                 \
      }                                                                                                 \
      return api_ret;                                                                                   \
    };                                                                                                  \
    at_npu::native::OpCommand::RunOpApiV2(#aclnn_api, acl_call);                                        \
    if (unInitMemFunc) {                                                                                \
      unInitMemFunc(nullptr, false);                                                                    \
    }                                                                                                   \
    UnInitCacheThreadLocal();                                                                           \
  } while (false)

#define EXEC_NPU_CMD_SCATTER_V2(aclnn_api, ...)                                                                               \
  do {                                                                                                                \
    static const auto getWorkspaceSizeFuncAddr = GetOpApiFuncAddr(#aclnn_api "GetWorkspaceSize");                     \
    static const auto opApiFuncAddr = GetOpApiFuncAddr(#aclnn_api);                                                   \
    static const auto initMemAddr = GetOpApiFuncAddr("InitHugeMemThreadLocal");                                       \
    static const auto unInitMemAddr = GetOpApiFuncAddr("UnInitHugeMemThreadLocal");                                   \
    static const auto releaseMemAddr = GetOpApiFuncAddr("ReleaseHugeMem");                                            \
    TORCH_CHECK(                                                                                                      \
        getWorkspaceSizeFuncAddr != nullptr && opApiFuncAddr != nullptr,                                              \
        #aclnn_api,                                                                                                   \
        " or ",                                                                                                       \
        #aclnn_api "GetWorkspaceSize",                                                                                \
        " not in ",                                                                                                   \
        GetOpApiLibName(),                                                                                            \
        ", or ",                                                                                                      \
        GetOpApiLibName(),                                                                                            \
        " not found.",                                                                                                \
        OPS_ERROR(ErrCode::PTR));                                                                                     \
    auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);                                                   \
    if (c10_npu::check_enqueue_need_use(acl_stream)) {                                                                \
      c10_npu::UseStreamResInCurrentThread(acl_stream);                                                               \
    }                                                                                                                 \
    OP_EXEC_LOG_WITH_TASK_QUEUE(#aclnn_api, "EXEC_NPU_CMD", "2", acl_stream, __VA_ARGS__);                            \
    auto copied_params = CopyTypesV2(__VA_ARGS__);                                                                    \
    uint32_t aic_num = UINT32_MAX;                                                                                    \
    uint32_t aiv_num = UINT32_MAX;                                                                                    \
    if (c10_npu::is_core_control_enabled()) {                                                                         \
      aic_num = c10_npu::GetResInCurrentThread(c10_npu::acl::ACL_RT_DEV_RES_CUBE_CORE);                               \
      aiv_num = c10_npu::GetResInCurrentThread(c10_npu::acl::ACL_RT_DEV_RES_VECTOR_CORE);                             \
    }                                                                                                                 \
    auto snapshot = CaptureScatterDeterministicSnapshot();                                                          \
    auto acl_call = [copied_params, acl_stream, aic_num, aiv_num, snapshot]() -> int {                                \
      if (c10_npu::check_dequeue_need_use(acl_stream)) {                                                              \
        c10_npu::UseStreamResInCurrentThread(acl_stream);                                                             \
      }                                                                                                               \
      uint64_t workspace_size = 0;                                                                                    \
      uint64_t* workspace_size_addr = &workspace_size;                                                                \
      aclOpExecutor* executor = nullptr;                                                                              \
      aclOpExecutor** executor_addr = &executor;                                                                      \
      InitHugeMemThreadLocal initMemFunc = reinterpret_cast<InitHugeMemThreadLocal>(initMemAddr);                     \
      UnInitHugeMemThreadLocal unInitMemFunc = reinterpret_cast<UnInitHugeMemThreadLocal>(unInitMemAddr);             \
      int api_ret = 0;                                                                                                \
      if (hit_cache_v2(acl_stream, #aclnn_api, opApiFuncAddr, copied_params, &api_ret, aic_num, aiv_num, snapshot)) { \
        return api_ret;                                                                                               \
      }                                                                                                               \
      if (initMemFunc) {                                                                                              \
        initMemFunc(nullptr, false);                                                                                  \
      }                                                                                                               \
      auto converted_params = ConvertTypesV2(copied_params, workspace_size_addr, executor_addr);                      \
      auto getWorkspaceSizeFunc = ConvertToOpApiFunc(converted_params, getWorkspaceSizeFuncAddr);                     \
      auto workspace_status = call(getWorkspaceSizeFunc, converted_params);                                           \
      NPU_CHECK_ERROR(workspace_status, "call " #aclnn_api " failed");                                                \
      void* workspace_addr = nullptr;                                                                                 \
      at::Tensor workspace_tensor;                                                                                    \
      if (workspace_size != 0) {                                                                                      \
        workspace_tensor = at_npu::native::OpPreparation::unsafe_empty_workspace(workspace_size, acl_stream);         \
        workspace_addr = const_cast<void*>(workspace_tensor.storage().data());                                        \
      }                                                                                                               \
      OpApiFunc opApiFunc = reinterpret_cast<OpApiFunc>(opApiFuncAddr);                                               \
      api_ret = opApiFunc(workspace_addr, workspace_size, executor, acl_stream);                                      \
      NPU_CHECK_ERROR(api_ret, "call " #aclnn_api " failed");                                                         \
      ReleaseConvertTypes(converted_params);                                                                          \
      ReleaseHugeMem releaseMemFunc = reinterpret_cast<ReleaseHugeMem>(releaseMemAddr);                               \
      if (releaseMemFunc) {                                                                                           \
        releaseMemFunc(nullptr, false);                                                                               \
      }                                                                                                               \
      if (unInitMemFunc) {                                                                                            \
        unInitMemFunc(nullptr, false);                                                                                \
      }                                                                                                               \
      UnInitCacheThreadLocal();                                                                                       \
      return api_ret;                                                                                                 \
    };                                                                                                                \
    at_npu::native::OpCommand::RunOpApiV2(#aclnn_api, acl_call);                                                      \
  } while (false)

#define EXEC_NPU_CMD_SCATTER(aclnn_api, ...)                                                      \
  do {                                                                                    \
    const auto task_queue_enable = c10_npu::option::OptionsManager::GetTaskQueueEnable(); \
    if (task_queue_enable == 2) {                                                         \
      EXEC_NPU_CMD_SCATTER_V2(aclnn_api, __VA_ARGS__);                                            \
    } else {                                                                              \
      EXEC_NPU_CMD_SCATTER_V1(aclnn_api, __VA_ARGS__);                                            \
    }                                                                                     \
  } while (false)

} // namespace scatter_dispatch

void scatter_pa_kv_cache_cached(
    const at::Tensor& key,
    const at::Tensor& value,
    at::Tensor& key_cache,
    at::Tensor& value_cache,
    const at::Tensor& slot_mapping,
    c10::string_view cache_mode,
    c10::string_view scatter_mode)
{
    std::string cache_mode_str(cache_mode);
    std::string scatter_mode_str(scatter_mode);
    char* cache_mode_ptr = cache_mode_str.data();
    char* scatter_mode_ptr = scatter_mode_str.data();
    c10::optional<at::Tensor> optional_tensor = c10::nullopt;
    const std::array<int64_t, 2> strides_size = {1, 1};
    const std::array<int64_t, 2> offsets_size = {0, 0};
    at::IntArrayRef strides(strides_size);
    at::IntArrayRef offsets(offsets_size);

    // Use the local ordinary PTA dispatcher; retain strided cache views.
    using namespace scatter_dispatch;
    EXEC_NPU_CMD_SCATTER(aclnnScatterPaKvCache,
                     key, key_cache, slot_mapping, value, value_cache,
                     optional_tensor, optional_tensor, optional_tensor,
                     cache_mode_ptr, scatter_mode_ptr, strides, offsets);
}

#undef SCATTER_MEMCPY_TO_BUF
#undef EXEC_NPU_CMD_SCATTER
#undef EXEC_NPU_CMD_SCATTER_V1
#undef EXEC_NPU_CMD_SCATTER_V2
} // namespace vllm_ascend

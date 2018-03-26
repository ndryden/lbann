////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2016, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory.
// Written by the LBANN Research Team (B. Van Essen, et al.) listed in
// the CONTRIBUTORS file. <lbann-dev@llnl.gov>
//
// LLNL-CODE-697807.
// All rights reserved.
//
// This file is part of LBANN: Livermore Big Artificial Neural Network
// Toolkit. For details, see http://software.llnl.gov/LBANN or
// https://github.com/LLNL/LBANN.
//
// Licensed under the Apache License, Version 2.0 (the "Licensee"); you
// may not use this file except in compliance with the License.  You may
// obtain a copy of the License at:
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
// implied. See the License for the specific language governing
// permissions and limitations under the license.
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_LAYER_POOLING_HPP_INCLUDED
#define LBANN_LAYER_POOLING_HPP_INCLUDED

#include <utility>
#include <vector>
#include "lbann/layers/transform/transform.hpp"
#include "lbann/utils/cudnn_wrapper.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/im2col.hpp"
#include "lbann_config.hpp"
#include "lbann/distconv.hpp"

namespace lbann {

// Forward declaration
template <data_layout T_layout>
class unpooling_layer;

/** Pooling layer. */
template <data_layout T_layout = data_layout::DATA_PARALLEL>
class pooling_layer : public transform_layer {
 private:

  /// Pooling mode
  const pool_mode m_pool_mode;

  /// Pooling window dimensions
  std::vector<int> m_pool_dims;
  /// Size of pooling window
  int m_pool_size;
  /// Pooling padding
  std::vector<int> m_pads;
  /// Pooling strides
  std::vector<int> m_strides;
 
  /** Input indices for max pooling.
   *  Each entry corresponds to a local entry in the activations
   *  matrix. The entry gives the index of the maximum entry within
   *  the pooling window.
   */
  std::vector<int> m_max_pool_indices;

#ifdef LBANN_HAS_CUDNN
  /// Pooling descriptor
  cudnnPoolingDescriptor_t m_pooling_cudnn_desc;
#endif // LBANN_HAS_CUDNN

  friend class unpooling_layer<T_layout>;

 public:

  pooling_layer(lbann_comm *comm,
                int num_data_dims,
                int pool_dim,
                int pad,
                int stride,
                pool_mode mode,
                cudnn::cudnn_manager *cudnn = nullptr)
    : pooling_layer(comm,
                    num_data_dims,
                    std::vector<int>(num_data_dims, pool_dim),
                    std::vector<int>(num_data_dims, pad),
                    std::vector<int>(num_data_dims, stride),
                    mode,
                    cudnn) {}

  pooling_layer(lbann_comm *comm,
                int num_data_dims,
                std::vector<int> pool_dims,
                std::vector<int> pads,
                std::vector<int> strides,
                pool_mode mode,
                cudnn::cudnn_manager *cudnn = nullptr)
    : transform_layer(comm),
      m_pool_mode(mode),
      m_pool_dims(pool_dims),
      m_pads(pads),
      m_strides(strides) {
    static_assert(T_layout == data_layout::DATA_PARALLEL,
                  "pooling only supports DATA_PARALLEL");

    // Initialize input dimensions and pooling parameters
    m_pool_size = std::accumulate(m_pool_dims.begin(),
                                  m_pool_dims.end(),
                                  1,
                                  std::multiplies<int>());

  #ifdef LBANN_HAS_CUDNN

    // Initialize cuDNN objects
    m_pooling_cudnn_desc = nullptr;

    // Initialize GPU memory if using GPU
    if (cudnn) {
      this->m_using_gpus = true;
      this->m_cudnn = cudnn;
    }
  #endif // LBANN_HAS_CUDNN

  }

  pooling_layer(const pooling_layer& other) :
    transform_layer(other),
    m_pool_mode(other.m_pool_mode),
    m_pool_dims(other.m_pool_dims),
    m_pool_size(other.m_pool_size),
    m_pads(other.m_pads),
    m_strides(other.m_strides),
    m_max_pool_indices(other.m_max_pool_indices) {
  #ifdef LBANN_HAS_CUDNN
    m_pooling_cudnn_desc = nullptr;
    cudnn::copy_pooling_cudnn_desc(other.m_pooling_cudnn_desc, m_pooling_cudnn_desc);
  #endif // LBANN_HAS_CUDNN
  }

  pooling_layer& operator=(const pooling_layer& other){
    transform_layer::operator=(other);
    m_pool_mode = other.m_pool_mode;
    m_pool_dims = other.m_pool_dims;
    m_pool_size = other.m_pool_size;
    m_pads = other.m_pads;
    m_strides = other.m_strides;
    m_max_pool_indices = other.m_max_pool_indices;
  #ifdef LBANN_HAS_CUDNN
    cudnn::copy_pooling_cudnn_desc(other.m_pooling_cudnn_desc, m_pooling_cudnn_desc);
  #endif // LBANN_HAS_CUDNN
    return *this;
  }
    
  pooling_layer* copy() const override { return new pooling_layer(*this); }
  std::string get_type() const override { return "pooling"; }
  data_layout get_data_layout() const override { return T_layout; }

  /** Returns description of ctor params */
  std::string get_description() const override {
    std::stringstream s;
    s << " pooling; num_data_dims: "
    + std::to_string(m_pool_dims.size()) + " pool_dims: ";
    for (size_t h=0; h<this->m_pool_dims.size(); h++) {
      s << this->m_pool_dims[h] << " ";
    }
    s << " pads: ";
    for (size_t h=0; h<this->m_pads.size(); h++) {
      s << this->m_pads[h] << " ";
    }
    s << " strides: ";
    for (size_t h=0; h<this->m_strides.size(); h++) {
      s << this->m_strides[h] << " ";
    }
    s << " pool_mode: " << get_pool_mode_name(this->m_pool_mode);
    s << " dataLayout: " << this->get_data_layout_string(get_data_layout());
    return s.str();
  }

  /// Destructor
  ~pooling_layer() override {
  #ifdef LBANN_HAS_CUDNN
    // Destroy cuDNN objects
    if (m_pooling_cudnn_desc) {
      CHECK_CUDNN(cudnnDestroyPoolingDescriptor(m_pooling_cudnn_desc));
    }
  #endif // LBANN_HAS_CUDNN
  }

  void setup_dims() override {

    // Initialize previous neuron tensor dimensions
    transform_layer::setup_dims();

    // Initialize neuron tensor dimensions
    for(int i=0; i<this->m_num_neuron_dims-1; ++i) {
      const int effective_dim = (this->m_prev_neuron_dims[i+1]
                                 + 2 * m_pads[i] - m_pool_dims[i] + 1);
      this->m_neuron_dims[i+1] = ((effective_dim + m_strides[i] - 1)
                                  / m_strides[i]);
    }
    this->m_num_neurons = std::accumulate(this->m_neuron_dims.begin(),
                                          this->m_neuron_dims.end(),
                                          1,
                                          std::multiplies<int>());

  }

  /// Initialize GPU objects
  void setup_gpu() override {
    transform_layer::setup_gpu();
  #ifndef LBANN_HAS_CUDNN
    throw lbann_exception("lbann_layer_pooling: cuDNN not detected");
  #else

    // Set pooling descriptor
    cudnnPoolingMode_t cudnn_pool_mode;
    switch(m_pool_mode) {
    case pool_mode::max:
      cudnn_pool_mode = CUDNN_POOLING_MAX; break;
    case pool_mode::average:
      cudnn_pool_mode = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING; break;
    case pool_mode::average_no_pad:
      cudnn_pool_mode = CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING; break;
    default:
      throw lbann_exception("pooling_layer: no GPU implementation for pooling mode");
    }
    CHECK_CUDNN(cudnnCreatePoolingDescriptor(&m_pooling_cudnn_desc));
    CHECK_CUDNN(cudnnSetPoolingNdDescriptor(m_pooling_cudnn_desc,
                                            cudnn_pool_mode,
                                            CUDNN_PROPAGATE_NAN,
                                            m_pool_dims.size(),
                                            m_pool_dims.data(),
                                            m_pads.data(),
                                            m_strides.data()));

#ifdef LBANN_HAS_DISTCONV
    setup_tensors();
#endif
  #endif // #ifndef LBANN_HAS_CUDNN
  }

  protected:

  void fp_compute() override {
    if(this->m_using_gpus) {
#ifdef LBANN_HAS_DISTCONV
      if (m_distconv_enabled) {
        fp_compute_distconv();
        //MPI_Barrier(MPI_COMM_WORLD);
        //exit(0);
      } else {
        fp_compute_cudnn();
      }
#else      
      fp_compute_cudnn();
#endif      
    } else {
      fp_compute_im2col();
    }
  }

  void bp_compute() override {
    if(this->m_using_gpus) {
#ifdef LBANN_HAS_DISTCONV
      if (m_distconv_enabled) {
        bp_compute_distconv();        
      } else {
        bp_compute_cudnn();        
      }
#else
      bp_compute_cudnn();
#endif      
    } else {
      bp_compute_im2col();
    }
  }

 private:

  /// Pooling forward propagation with cuDNN
  void fp_compute_cudnn() {
  #ifndef LBANN_HAS_CUDNN
    throw lbann_exception("pooling_layer: cuDNN not detected");
  #else

    // Useful constants
    const DataType one = 1;
    const DataType zero = 0;

    // Perform pooling with each GPU
    const int num_gpus = this->m_cudnn->get_num_gpus();
    for(int i=0; i<num_gpus; ++i) {
      CHECK_CUDA(cudaSetDevice(this->m_cudnn->get_gpu(i)));
      CHECK_CUDNN(cudnnSetStream(this->m_cudnn->get_handle(i),
                                 this->m_cudnn->get_stream(i)));
      CHECK_CUDNN(cudnnPoolingForward(this->m_cudnn->get_handle(i),
                                      m_pooling_cudnn_desc,
                                      &one,
                                      this->m_prev_activations_cudnn_desc,
                                      this->m_prev_activations_d[0].get_locked_data(i),
                                      &zero,
                                      this->m_activations_cudnn_desc,
                                      this->m_activations_d[0].get_data(i)));
    }

  #endif // #ifndef LBANN_HAS_CUDNN
  }

  /// Pooling backward propagation with cuDNN
  void bp_compute_cudnn() {    
  #ifndef LBANN_HAS_CUDNN
    throw lbann_exception("pooling_layer: cuDNN not detected");
  #else

    // Useful constants
    const DataType one = DataType(1);

    // Get number of GPUs
    const int num_gpus = this->m_cudnn->get_num_gpus();

    // Perform back propagation on each GPU
    for(int i=0; i<num_gpus; ++i) {
      CHECK_CUDA(cudaSetDevice(this->m_cudnn->get_gpu(i)));
      CHECK_CUDNN(cudnnSetStream(this->m_cudnn->get_handle(i),
                                 this->m_cudnn->get_stream(i)));
      CHECK_CUDNN(cudnnPoolingBackward(this->m_cudnn->get_handle(i),
                                       m_pooling_cudnn_desc,
                                       &one,
                                       this->m_activations_cudnn_desc,
                                       this->m_activations_d[0].get_locked_data(i),
                                       this->m_prev_error_signals_cudnn_desc,
                                       this->m_prev_error_signals_d[0].get_locked_data(i),
                                       this->m_prev_activations_cudnn_desc,
                                       this->m_prev_activations_d[0].get_locked_data(i),
                                       &one,
                                       this->m_error_signals_cudnn_desc,
                                       this->m_error_signals_d[0].get_data(i)));
    }

  #endif // #ifndef LBANN_HAS_CUDNN
  }

  /// Pooling forward propagation with im2col
  void fp_compute_im2col() {

    // Throw exception if pooling mode is not max or average pooling
    if(m_pool_mode != pool_mode::max
       && m_pool_mode != pool_mode::average) {
      throw lbann_exception("pooling_layer: CPU pooling layer only implements max and average pooling");
    }

    // Local matrices
    const auto& local_input = get_local_prev_activations();
    auto& local_output = get_local_activations();

    // Pool parameters
    const int local_width = local_input.Width();
    const int num_channels = this->m_prev_neuron_dims[0];
    const int num_per_output_channel = this->m_num_neurons / num_channels;

    // Initialize max pool indices if needed
    if(m_pool_mode == pool_mode::max) {
      m_max_pool_indices.assign(this->m_num_neurons * local_width, 0);
    }

    // Initialize matrices
    Mat im2col_mat(m_pool_size * num_channels, num_per_output_channel);
    Mat input_mat;

    // Iterate through data samples
    for(int sample = 0; sample < local_width; ++sample) {

      // Construct im2col matrix from input
      El::LockedView(input_mat, local_input,
                     El::ALL, El::IR(sample));
      im2col(input_mat,
             im2col_mat,
             num_channels,
             this->m_num_prev_neuron_dims - 1,
             &this->m_prev_neuron_dims[1],
             m_pads.data(),
             m_pool_dims.data(),
             m_strides.data());

      if(m_pool_mode == pool_mode::max) {
        // Apply max pooling
        DataType *output_buffer = local_output.Buffer(0, sample);
        int *indices_buffer = &m_max_pool_indices[sample * this->m_num_neurons];
        #pragma omp parallel for
        for(int channel = 0; channel < num_channels; ++channel) {
          for(int j = 0; j < num_per_output_channel; ++j) {
            DataType *im2col_buffer = im2col_mat.Buffer(channel*m_pool_size, j);
            DataType max_entry = im2col_buffer[0];
            int max_index = 0;
            for(int i = 1; i < m_pool_size; ++i) {
              const DataType current_entry = im2col_buffer[i];
              if(current_entry > max_entry) {
                max_entry = current_entry;
                max_index = i;
              }
            }
            const int output_index = j + channel * num_per_output_channel;
            output_buffer[output_index] = max_entry;
            indices_buffer[output_index] = max_index;
          }
        }
      }

      if(m_pool_mode == pool_mode::average) {
        // Apply average pooling
        DataType *output_buffer = local_output.Buffer(0, sample);
        #pragma omp parallel for
        for(int channel = 0; channel < num_channels; ++channel) {
          for(int j = 0; j < num_per_output_channel; ++j) {
            const DataType *im2col_buffer
              = im2col_mat.LockedBuffer(channel*m_pool_size, j);
            DataType output_entry = 0;
            for(int i = 0; i < m_pool_size; ++i) {
              output_entry += im2col_buffer[i];
            }
            output_entry /= m_pool_size;
            const int output_index = j + channel * num_per_output_channel;
            output_buffer[output_index] = output_entry;
          }
        }
      }

    }

  }

  /// Pooling forward propagation with im2col
  void bp_compute_im2col() {

    // Throw exception if pooling mode is not max or average pooling
    if(m_pool_mode != pool_mode::max
        && m_pool_mode != pool_mode::average) {
      throw lbann_exception("pooling_layer: CPU pooling layer only implements max and average pooling");
    }

    // Local matrices
    const auto& local_gradient_wrt_output = get_local_prev_error_signals();
    auto& local_gradient_wrt_input = get_local_error_signals();

    // Pool parameters
    const int input_size = local_gradient_wrt_input.Height();
    const int local_width = local_gradient_wrt_output.Width();
    const int num_channels = this->m_prev_neuron_dims[0];
    const int num_per_input_channel = this->m_num_neurons / num_channels;

    // Initialize matrices
    Mat im2col_mat(m_pool_size * num_channels, num_per_input_channel);
    Mat gradient_wrt_input_col(input_size, 1);

    // Iterate through data samples
    for(int sample = 0; sample < local_width; ++sample) {

      // Compute gradient w.r.t. im2col matrix for max pooling
      if(m_pool_mode == pool_mode::max) {

        // Clear im2col matrix
        El::Zero(im2col_mat);

        // Copy previous error signal to im2col matrix entries
        // corresponding to max
        const DataType *gradient_wrt_output_buffer
          = local_gradient_wrt_output.LockedBuffer(0, sample);
        const int *indices_buffer
          = &m_max_pool_indices[sample * this->m_num_neurons];
        #pragma omp parallel for
        for(int channel = 0; channel < num_channels; ++channel) {
          for(int j = 0; j < num_per_input_channel; ++j) {
            const int input_index = j + channel * num_per_input_channel;
            const int max_index = indices_buffer[input_index];
            DataType *im2col_buffer = im2col_mat.Buffer(channel*m_pool_size, j);
            im2col_buffer[max_index]
              = gradient_wrt_output_buffer[input_index];
          }
        }

      }

      // Compute gradient w.r.t. im2col matrix for average pooling
      if(m_pool_mode == pool_mode::average) {
        const DataType *gradient_wrt_output_buffer
          = local_gradient_wrt_output.LockedBuffer(0, sample);
        #pragma omp parallel for
        for(int channel = 0; channel < num_channels; ++channel) {
          for(int j = 0; j < num_per_input_channel; ++j) {
            DataType *im2col_buffer = im2col_mat.Buffer(channel*m_pool_size, j);
            const int input_index = j + channel * num_per_input_channel;
            const DataType output_entry
              = gradient_wrt_output_buffer[input_index] / m_pool_size;
            for(int i = 0; i < m_pool_size; ++i) {
              im2col_buffer[i] = output_entry;
            }
          }
        }

      }

      // Compute error signal (i.e. gradient w.r.t. input)
      col2im(im2col_mat,
             gradient_wrt_input_col,
             num_channels,
             this->m_num_prev_neuron_dims - 1,
             &this->m_prev_neuron_dims[1],
             m_pads.data(),
             m_pool_dims.data(),
             m_strides.data());
      local_gradient_wrt_input(El::ALL, El::IR(sample)) += gradient_wrt_input_col;

    }

  }
  
  void setup_tensors() {
#ifndef LBANN_HAS_DISTCONV
    throw lbann_exception(
        std::string {} + __FILE__ + " " + std::to_string(__LINE__) + " :: " +
        "Layer: DISTCONV not detected");
#else
    MPIPrintStreamDebug()
        << "pooling: setup_tensors."
        << " pads: " << m_pads[0] << "x" << m_pads[1]
        << ", pool_dims: " << m_pool_dims[0] << "x" << m_pool_dims[1]
        << ", m_strides: " << m_strides[0] << "x" << m_strides[1]
        << "\n";

    if (!(m_pads[0] == 0 && m_pads[1] == 0 &&
          m_pool_dims[0] % 2 != 0 && m_pool_dims[1] % 2 != 0)) {
      MPIPrintStreamDebug() << "pooling: unsupported \n";
      return;
    }
    
    int stencil_h = (m_pool_dims[0] - 1) / 2;
    int stencil_w = (m_pool_dims[1] - 1) / 2;

    if (!((m_strides[0] == 1 && m_strides[1] == 1) ||
         (m_strides[0] == stencil_h + 1 &&
          m_strides[1] == stencil_w + 1))) {
      MPIPrintStreamDebug() << "pooling: unsupported \n";
      return;
    }

    // REFACTORING: duplicated at convolution::setup_tensors
    Array4 input_tensor_shape =
        {m_prev_neuron_dims[2], m_prev_neuron_dims[1],
         m_prev_neuron_dims[0],
         this->m_model->get_max_mini_batch_size()};

    // shape dim must be divisible by strides
    if (!(input_tensor_shape[0] % m_strides[0] == 0 &&
          input_tensor_shape[1] % m_strides[1] == 0)) {
      MPIPrintStreamDebug() << "Not divisible by strides\n";
      return;
    }
    
    m_distconv_enabled = true;

    Array4 input_local_shape = input_tensor_shape;
    // Assuming single GPU per rank
    input_local_shape[3] = m_max_mini_batch_size_per_gpu;
    Array4 division_block_size = {1, 1, 1, 1};

    LocaleMPI loc(m_comm->get_model_comm().comm);
    // Sample distribution
    Array4 sample_decomposition = {1, 1, 1, m_comm->get_procs_per_model()};
    
    m_prev_activations_e = ConstTensorDev(input_tensor_shape, loc,
                                          Dist(sample_decomposition),
                                          input_local_shape,
                                          division_block_size);

    assert0(dc::tensor::View(
        m_prev_activations_e,
        m_prev_activations_d[0].get_locked_data(0)));

    Array4 spatial_decomposition = {1, m_comm->get_procs_per_model(), 1, 1};
    Array4 overlap = {stencil_w, stencil_h, 0, 0};
    Array4 spatial_block_size = {m_strides[1], m_strides[0], 1, 1};
    Array4 spatial_local_size = {0, 0, 0, 0};

    m_prev_activations_t = TensorDev(input_tensor_shape, loc,
                                     Dist(spatial_decomposition, overlap),
                                     spatial_local_size, spatial_block_size);
    m_prev_activations_t.allocate();
    m_prev_activations_t.zero();
    
    Array4 output_tensor_shape = {m_neuron_dims[2], m_neuron_dims[1],
                                  m_neuron_dims[0],
                                  this->m_model->get_max_mini_batch_size()};
    Array4 output_local_shape = output_tensor_shape;
    output_local_shape[3] = m_max_mini_batch_size_per_gpu;

    m_activations_t = TensorDev(output_tensor_shape,
                                loc, Dist(spatial_decomposition));

    m_activations_t.allocate();

    m_activations_e = TensorDev(output_tensor_shape, loc,
                                Dist(sample_decomposition),
                                output_local_shape,
                                division_block_size);
    // prev_error_signals
    m_prev_error_signals_t = TensorDev(output_tensor_shape, loc,
                                       Dist(spatial_decomposition));
    m_prev_error_signals_t.allocate();
    m_prev_error_signals_t.zero();
    m_prev_error_signals_e = ConstTensorDev(output_tensor_shape, loc,
                                            Dist(sample_decomposition),
                                            output_local_shape,
                                            division_block_size);
    assert0(dc::tensor::View(
        m_prev_error_signals_e,
        m_prev_error_signals_d[0].get_locked_data(0)));

    // error_signals
    m_error_signals_e = TensorDev(input_tensor_shape, loc,
                                  Dist(sample_decomposition),
                                  input_local_shape, division_block_size);
    // Needs halo
    m_error_signals_t = TensorDev(input_tensor_shape, loc,
                                  Dist(spatial_decomposition, overlap),
                                  spatial_local_size, spatial_block_size);
    m_error_signals_t.allocate();

    // Init the dc::Pooling layer
    m_pooling = new dc::Pooling<dc::cudnn::BackendCUDNN>(
        *this->m_cudnn->get_distconv_backend());

    std::string mode;
    switch(m_pool_mode) {
      case pool_mode::max:
        mode = "MAX"; break;
      case pool_mode::average:
        mode = "AVERAGE"; break;
      case pool_mode::average_no_pad:
        mode = "AVERAGE_NO_PAD"; break;
    default:
      throw lbann_exception("pooling_layer: no DISTCONV implementation for pooling mode");
    }
    
    m_pooling->setup(m_prev_activations_t,
                     m_activations_t,
                     m_pool_dims[0], m_pool_dims[1],
                     m_pads[0], m_pads[1],
                     m_strides[0], m_strides[1],
                     mode);

    MPIPrintStreamDebug()
        << "Pooling. "
        << "prev_activations_e: " << m_prev_activations_e
        << ", prev_activations_t: " << m_prev_activations_t
        << ", activations_e: " << m_activations_e
        << ", activations_t: " << m_activations_t
        << ", prev_error_signals_e: " << m_prev_activations_e
        << ", prev_error_signals_t: " << m_prev_activations_t
        << ", error_signals_e: " << m_error_signals_e
        << ", error_signals_t: " << m_error_signals_t
        << "\n";
#endif
  }
  
  void fp_compute_distconv() {
#ifndef LBANN_HAS_DISTCONV
    throw lbann_exception("pooling_layer: DISTCONV not detected");
#else
    MPIPrintStreamDebug() << "Forward pooling\n";

    assert_always(m_distconv_enabled);
    
    assert0(dc::tensor::View(
        m_prev_activations_e,
        m_prev_activations_d[0].get_locked_data(0)));
    assert0(dc::tensor::Copy(
        m_prev_activations_t, m_prev_activations_e));

    m_pooling->set_num_samples(this->m_model->get_current_mini_batch_size());

    m_pooling->forward(1.0, m_prev_activations_t,
                       0.0, m_activations_t);

    assert0(dc::tensor::View(
        m_activations_e, m_activations_d[0].get_data(0)));
    assert0(dc::tensor::Copy(
        m_activations_e, m_activations_t));

#endif
  }

  void bp_compute_distconv() {
#ifndef LBANN_HAS_DISTCONV
    throw lbann_exception("pooling_layer: DISTCONV not detected");
#else
    MPIPrintStreamDebug() << "Backward pooling\n";

    assert_always(m_distconv_enabled);

    assert0(dc::tensor::View(
        m_prev_error_signals_e,
        m_prev_error_signals_d[0].get_locked_data(0)));
    assert0(dc::tensor::Copy(
        m_prev_error_signals_t, m_prev_error_signals_e));

    assert0(dc::tensor::View(
        m_error_signals_e, m_error_signals_d[0].get_data(0)));
    assert0(dc::tensor::Copy(
        m_error_signals_t, m_error_signals_e));

    m_pooling->backward(1.0, m_activations_t, m_prev_error_signals_t,
                        m_prev_activations_t, 1.0, m_error_signals_t);

    assert0(dc::tensor::Copy(
        m_error_signals_e, m_error_signals_t));
#endif    
  }  
  
#ifdef LBANN_HAS_DISTCONV
  bool m_distconv_enabled = false;
  dc::Pooling<dc::cudnn::BackendCUDNN> *m_pooling;
  // Forward prop
  TensorDev m_prev_activations_t;
  ConstTensorDev m_prev_activations_e;
  TensorDev m_activations_t;
  TensorDev m_activations_e;
  // Backward prop
  TensorDev m_prev_error_signals_t;
  ConstTensorDev m_prev_error_signals_e;
  TensorDev m_error_signals_t;
  TensorDev m_error_signals_e;
  
#endif

};

} // namespace lbann

#endif // LBANN_LAYER_POOLING_HPP_INCLUDED

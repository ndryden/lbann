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

#ifndef LBANN_LAYER_ACTIVATION_RELU_HPP_INCLUDED
#define LBANN_LAYER_ACTIVATION_RELU_HPP_INCLUDED

#include "lbann/layers/activations/activation.hpp"
#include "lbann/utils/cudnn_wrapper.hpp"
#include "lbann_config.hpp"
#include "lbann/distconv.hpp"

namespace lbann {

/** Rectified linear unit activation function.
 *  See https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
 */
template <data_layout T_layout>
class relu_layer : public entrywise_activation_layer {

 private:

#ifdef LBANN_HAS_CUDNN
  /** Activation descriptor. */
  cudnnActivationDescriptor_t m_activation_cudnn_desc;
#endif // LBANN_HAS_CUDNN

 public:
  relu_layer(lbann_comm *comm,
             cudnn::cudnn_manager *cudnn = nullptr)
    : entrywise_activation_layer(comm) {
  #ifdef LBANN_HAS_CUDNN
    m_activation_cudnn_desc = nullptr;
    this->m_cudnn = cudnn;
    if (this->m_cudnn) {
      this->m_using_gpus = true;
    }
  #endif // LBANN_HAS_CUDNN
  }

  relu_layer(const relu_layer& other) :
    entrywise_activation_layer(other) {
  #ifdef LBANN_HAS_CUDNN
    m_activation_cudnn_desc = nullptr;
    cudnn::copy_activation_cudnn_desc(other.m_activation_cudnn_desc,
                                      m_activation_cudnn_desc);
  #endif // LBANN_HAS_CUDNN
  }

  relu_layer& operator=(const relu_layer& other) {
    entrywise_activation_layer::operator=(other);
  #ifdef LBANN_HAS_CUDNN
    cudnn::copy_activation_cudnn_desc(other.m_activation_cudnn_desc,
                                      m_activation_cudnn_desc);
  #endif // LBANN_HAS_CUDNN
    return *this;
  }

  ~relu_layer() override {
  #ifdef LBANN_HAS_CUDNN
    if (m_activation_cudnn_desc != nullptr) {
      CHECK_CUDNN(cudnnDestroyActivationDescriptor(m_activation_cudnn_desc));
    }
  #endif // LBANN_HAS_CUDNN
  }

  relu_layer* copy() const override { return new relu_layer(*this); }
  std::string get_type() const override { return "ReLU"; }

  /** Returns description of ctor params */
  std::string get_description() const override {
    return std::string {} +
     " relu" + " dataLayout: " + this->get_data_layout_string(get_data_layout());
  }

  data_layout get_data_layout() const override { return T_layout; }

  void setup_gpu() override {
    entrywise_activation_layer::setup_gpu();
  #ifndef LBANN_HAS_CUDNN
    throw lbann_exception("relu_layer: cuDNN not detected");
  #else
    CHECK_CUDNN(cudnnCreateActivationDescriptor(&m_activation_cudnn_desc));
    CHECK_CUDNN(cudnnSetActivationDescriptor(m_activation_cudnn_desc,
                                             CUDNN_ACTIVATION_RELU,
                                             CUDNN_PROPAGATE_NAN,
                                             0.0));

#ifdef LBANN_HAS_DISTCONV
    setup_tensors();
#endif
    
#endif // LBANN_HAS_CUDNN
  }

 protected:

  DataType activation(DataType x) const override {
    return x > DataType(0) ? x : DataType(0);
  }

  DataType activation_derivative(DataType x) const override {
    return x > DataType(0) ? DataType(1) : DataType(0);
  }

  void fp_compute_gpu() override {
  #ifndef LBANN_HAS_CUDNN
    throw lbann_exception("relu_layer: cuDNN not detected");
  #else
    // Useful constants
    const DataType one = 1;
    const DataType zero = 0;
#ifndef LBANN_HAS_DISTCONV
    // Apply activation on each GPU
    const int num_gpus = this->m_cudnn->get_num_gpus();
    for(int i = 0; i < num_gpus; ++i) {
      CHECK_CUDA(cudaSetDevice(this->m_cudnn->get_gpu(i)));
      CHECK_CUDNN(cudnnSetStream(this->m_cudnn->get_handle(i),
                                 this->m_cudnn->get_stream(i)));
      CHECK_CUDNN(cudnnActivationForward(this->m_cudnn->get_handle(i),
                                         m_activation_cudnn_desc,
                                         &one,
                                         this->m_prev_activations_cudnn_desc,
                                         this->m_prev_activations_d[0].get_locked_data(i),
                                         &zero,
                                         this->m_activations_cudnn_desc,
                                         this->m_activations_d[0].get_data(i)));
    }
#else
    assert0(dc::tensor::View(
        m_prev_activations_e,
        m_prev_activations_d[0].get_locked_data(0)));
    assert0(dc::tensor::View(
        m_activations_e, m_activations_d[0].get_data(0)));
    m_relu->set_num_samples(this->m_model->get_current_mini_batch_size());    
    m_relu->forward(one, m_prev_activations_e,
                    zero, m_activations_e);
#endif // LBANN_HAS_DISTCONV
  #endif // LBANN_HAS_CUDNN
  }

  void bp_compute_gpu() override {
  #ifndef LBANN_HAS_CUDNN
    throw lbann_exception("relu_layer: cuDNN not detected");
  #else
    // Useful constants
    const DataType one = 1;
#ifndef LBANN_HAS_DISTCONV
    // Apply activation derivative on each GPU
    const int num_gpus = this->m_cudnn->get_num_gpus();
    for(int i = 0; i < num_gpus; ++i) {
      CHECK_CUDA(cudaSetDevice(this->m_cudnn->get_gpu(i)));
      CHECK_CUDNN(cudnnSetStream(this->m_cudnn->get_handle(i),
                                 this->m_cudnn->get_stream(i)));
      CHECK_CUDNN(cudnnActivationBackward(this->m_cudnn->get_handle(i),
                                          m_activation_cudnn_desc,
                                          &one,
                                          this->m_prev_activations_cudnn_desc,
                                          this->m_prev_activations_d[0].get_locked_data(i),
                                          this->m_prev_error_signals_cudnn_desc,
                                          this->m_prev_error_signals_d[0].get_locked_data(i),
                                          this->m_activations_cudnn_desc,
                                          this->m_activations_d[0].get_locked_data(i),
                                          &one,
                                          this->m_error_signals_cudnn_desc,
                                          this->m_error_signals_d[0].get_data(i)));
    }
#else
    assert0(dc::tensor::View(
        m_activations_e, m_activations_d[0].get_data(0)));
    assert0(dc::tensor::View(
        m_prev_activations_e,
        m_prev_activations_d[0].get_locked_data(0)));
    assert0(dc::tensor::View(
        m_prev_error_signals_e,
        m_prev_error_signals_d[0].get_locked_data(0)));
    assert0(dc::tensor::View(
        m_error_signals_e,
        m_error_signals_d[0].get_data(0)));
    m_relu->backward(one, m_activations_e, m_prev_error_signals_e,
                     m_prev_activations_e, one,
                     m_error_signals_e);
#endif // LBANN_HAS_DISTCONV
  #endif // LBANN_HAS_CUDNN
  }

  void setup_tensors() {
#ifndef LBANN_HAS_DISTCONV
    throw lbann_exception(
        std::string {} + __FILE__ + " " + std::to_string(__LINE__) + " :: " +
        "Layer: DISTCONV not detected");
#else
    MPIPrintStreamDebug()
        << "relu: setup_tensors."
        << "\n";
    
    m_distconv_enabled = true;

    MPIPrintStreamDebug() << "relu: distconv enabled\n";

    // REFACTORING: duplicated at convolution::setup_tensors
    Array4 input_tensor_shape =
        {m_prev_neuron_dims[2], m_prev_neuron_dims[1],
         m_prev_neuron_dims[0],
         this->m_model->get_max_mini_batch_size()};

    Array4 input_local_shape = input_tensor_shape;
    // Assuming single GPU per rank
    input_local_shape[3] = m_max_mini_batch_size_per_gpu;
    Array4 division_block_size = {1, 1, 1, 1};

    LocaleMPI loc(m_comm->get_model_comm().comm);
    // Sample distribution
    Array4 sample_decomposition = {1, 1, 1, m_comm->get_procs_per_model()};

    // prev_activations
    m_prev_activations_e = ConstTensorDev(input_tensor_shape, loc,
                                          Dist(sample_decomposition),
                                          input_local_shape,
                                          division_block_size);
    assert0(dc::tensor::View(
        m_prev_activations_e,
        m_prev_activations_d[0].get_locked_data(0)));

    Array4 output_tensor_shape = {m_neuron_dims[2], m_neuron_dims[1],
                                  m_neuron_dims[0],
                                  this->m_model->get_max_mini_batch_size()};
    Array4 output_local_shape = output_tensor_shape;
    output_local_shape[3] = m_max_mini_batch_size_per_gpu;

    // activations
    m_activations_e = TensorDev(output_tensor_shape, loc,
                                Dist(sample_decomposition),
                                output_local_shape,
                                division_block_size);
    assert0(dc::tensor::View(
        m_activations_e, m_activations_d[0].get_data(0)));
    

    // prev_error_signals
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
    assert0(dc::tensor::View(
        m_error_signals_e,
        m_error_signals_d[0].get_data(0)));

    // Init the dc::Pooling layer
    m_relu = new dc::ReLU<dc::cudnn::BackendCUDNN>(
        *this->m_cudnn->get_distconv_backend());

    m_relu->setup(m_prev_activations_e,
                  m_activations_e,
                  m_error_signals_e,
                  m_prev_error_signals_e);
#endif
  }
  

#ifdef LBANN_HAS_DISTCONV
  bool m_distconv_enabled = false;
  dc::ReLU<dc::cudnn::BackendCUDNN> *m_relu;
  // Forward prop
  ConstTensorDev m_prev_activations_e;
  TensorDev m_activations_e;
  // Backward prop
  ConstTensorDev m_prev_error_signals_e;
  TensorDev m_error_signals_e;
#endif
  

};


} // namespace lbann

#endif // LBANN_LAYER_ACTIVATION_RELU_HPP_INCLUDED

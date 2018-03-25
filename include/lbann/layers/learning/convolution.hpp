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

#ifndef LBANN_LAYER_CONVOLUTION_HPP_INCLUDED
#define LBANN_LAYER_CONVOLUTION_HPP_INCLUDED

#include <vector>
#include "lbann/layers/learning/base_convolution.hpp"
#include "lbann/layers/layer.hpp"
#include "lbann/utils/cudnn_wrapper.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/random.hpp"
#include "lbann/utils/timer.hpp"
#include "lbann_config.hpp"

namespace lbann {

/// Convolution layer
template <data_layout T_layout = data_layout::DATA_PARALLEL>
class convolution_layer : public base_convolution_layer {
 private:

  friend class lbann_callback_imcomm;

 public:

  /// kernel tensor is output channels, input channels, conv dimension (w x h)
  /** Returns description of ctor params */
  std::string get_description() const override {
    std::stringstream s;
    s << " convolution; conv_dims: ";
    // for (size_t h=0; h<this->m_kernel_dims.size(); h++) {
    //   if (h == 0) { s << " channels (out x in) "; }
    //   if (h == 2) { s << " filters (w x h) "; }
    //   s << this->m_kernel_dims[h] << " ";
    // }
    s << get_topo_description();
    s << " pads: ";
    for (size_t h=0; h<this->m_pads.size(); h++) {
      s << this->m_pads[h] << " ";
    }
    s << " strides: ";
    for (size_t h=0; h<this->m_strides.size(); h++) {
      s << this->m_strides[h] << " ";
    }
    s << " num_output_channels: " << this->m_neuron_dims[0]
      << " has_bias: " << this->m_bias_scaling_factor
      << " dataLayout: " << this->get_data_layout_string(get_data_layout());
    return s.str();
  }

  std::string get_topo_description() const override {
    std::stringstream s;
    // Get the topo description from any parent class
    std::string str = base_convolution_layer::get_topo_description();
    s << str << " - ";

    // Display the topology of the kernel
    for (size_t h=0; h<this->m_kernel_dims.size(); h++) {
      if (h == 0) { s << "C="; }
      s << this->m_kernel_dims[h] ;
      if (h == 0) { s << "o,"; }
      if (h == 1) { s << "i F="; }
      if (this->m_kernel_dims.size() == 3) {
        if (h == 2) { s << "w "; }
      }else if (this->m_kernel_dims.size() == 4) {
        if (h == 2) { s << "w x "; }
        if (h == 3) { s << "h"; }
      }else {
        if (h > 1) {
          s << " ";
        }
      }
    }
    return s.str();;
  }

  convolution_layer(lbann_comm *comm,
                    int num_data_dims,
                    int num_output_channels,
                    int conv_dim,
                    int pad,
                    int stride,
                    bool has_bias = true,
                    cudnn::cudnn_manager *cudnn = nullptr)
      : convolution_layer(comm,
                          num_data_dims,
                          num_output_channels,
                          std::vector<int>(num_data_dims, conv_dim),
                          std::vector<int>(num_data_dims, pad),
                          std::vector<int>(num_data_dims, stride),
                          has_bias,
                          cudnn) {}

  convolution_layer(lbann_comm *comm,
                    int num_data_dims,
                    int num_output_channels,
                    std::vector<int> conv_dims,
                    std::vector<int> pads,
                    std::vector<int> strides,
                    bool has_bias = true,
                    cudnn::cudnn_manager *cudnn = nullptr)
      : base_convolution_layer(comm,
                               num_data_dims,
                               num_output_channels,
                               conv_dims,
                               pads,
                               strides,
                               has_bias,
                               cudnn) {
    static_assert(T_layout == data_layout::DATA_PARALLEL,
                  "convolution only supports DATA_PARALLEL");

    // Use GPUs if cuDNN manager is available
    if(this->m_cudnn) {
      this->m_using_gpus = true;
    }

  }

  convolution_layer* copy() const override { return new convolution_layer(*this); }

  std::string get_type() const override { return "convolution"; }

  data_layout get_data_layout() const override { return T_layout; }

  void setup_dims() override {

    // Initialize previous neuron tensor dimensions
    base_convolution_layer::setup_dims();

    // Initialize convolution kernel dimensions
    this->m_kernel_dims.insert(this->m_kernel_dims.begin() + 1,
                               this->m_prev_neuron_dims[0]);

    // Check if previous neuron tensor dimensions are valid
#ifdef LBANN_DEBUG
    if(this->m_num_neuron_dims != (int) this->m_kernel_dims.size() - 1) {
      throw lbann_exception("convolution_layer: neuron tensor dimensions are unexpected");
    }
#endif

    // Initialize neuron tensor dimensions
    this->m_neuron_dims[0] = this->m_kernel_dims[0];
    for(int i=0; i<this->m_num_neuron_dims-1; ++i) {
      const int effective_dim = (this->m_prev_neuron_dims[i+1]
                                 + 2 * this->m_pads[i]
                                 - this->m_kernel_dims[i+2] + 1);
      this->m_neuron_dims[i+1]= ((effective_dim + this->m_strides[i] - 1)
                                 / this->m_strides[i]);
    }
    this->m_num_neurons = std::accumulate(this->m_neuron_dims.begin(),
                                          this->m_neuron_dims.end(),
                                          1,
                                          std::multiplies<int>());

    // Get size of convolutional kernel
    this->m_kernel_size = std::accumulate(m_kernel_dims.begin(),
                                          m_kernel_dims.end(),
                                          1,
                                          std::multiplies<int>());

  }

  void setup_data() override {
    base_convolution_layer::setup_data();
    this->m_weights[0]->setup(m_kernel_dims);
    El::Zeros(this->m_kernel_gradient,
              this->m_weights[0]->get_matrix_height(),
              this->m_weights[0]->get_matrix_width());
  }

 protected:

  void fp_compute() override {
    if(this->m_using_gpus) {
#ifdef LBANN_HAS_DISTCONV
      if (m_distconv_enabled) {
        if (false) {
          apply_convolution_cudnn(true);        
          apply_convolution_distconv();
          MPI_Barrier(MPI_COMM_WORLD);
          exit(0);
          //apply_bias_distconv();        
          apply_bias_cudnn();
        } else {
          apply_convolution_distconv();
          apply_bias_distconv();
          // activations may be updated with bias, so its copy should
          // be done after applying bias
          MPIPrintStreamDebug() << "Copying back to sample parallel\n";
          assert0(dc::tensor::Copy(
              m_activations_e, m_activations_t));
        }
      } else {
        apply_convolution_cudnn(true);
        apply_bias_cudnn();
      }
#else
      apply_convolution_cudnn(true);
      apply_bias_cudnn();
#endif
    } else {
      apply_convolution_im2col(true);
      apply_bias_cpu();
    }
  }

  void bp_compute() override {
    if(this->m_using_gpus) {
#ifdef LBANN_HAS_DISTCONV
      if (m_distconv_enabled) {
        m_prev_error_signals_redistributed = false;
        
#ifdef DEBUG_COMPUTE_GRADIENTS
        compute_gradients_cudnn(false);
#endif
        if (true) {
          compute_gradients_distconv();
          apply_transposed_convolution_distconv();
        } else {
          compute_gradients_cudnn(false);
          apply_transposed_convolution_distconv();
          //apply_transposed_convolution_cudnn(false);
        }
      } else {
        compute_gradients_cudnn(false);
        apply_transposed_convolution_cudnn(false);
      }
#else
      compute_gradients_cudnn(false);
      apply_transposed_convolution_cudnn(false);
#endif
    } else {
      compute_gradients_im2col(false);
      apply_transposed_convolution_im2col(false);
    }
  }

  void setup_gpu() override {
    std::cerr << "setup gpu\n";
    base_convolution_layer::setup_gpu();
#ifdef LBANN_HAS_DISTCONV
    setup_tensors();
#endif
  }

  void setup_tensors() {
#ifndef LBANN_HAS_DISTCONV
    throw lbann_exception(
        std::string {} + __FILE__ + " " + std::to_string(__LINE__) + " :: " +
        "Layer: DISTCONV not detected");
#else
    MPIPrintStreamDebug() << "setup_tensors\n";

    std::stringstream ss;
    dc::util::print_vector(ss, m_kernel_dims.begin(), m_kernel_dims.end());
    MPIPrintStreamDebug()
        << "m_kernel_dims: " << ss.str() << "\n";

    if (!(m_kernel_dims[2] == m_kernel_dims[3] &&
          m_kernel_dims[2] == m_pads[0] * 2 + 1 &&
          m_kernel_dims[3] == m_pads[1] * 2 + 1)) {
      MPIPrintStreamDebug() << "Unsupported convolution\n";
      return;
    }

    m_distconv_enabled = true;
    
    Array4 input_tensor_shape = {m_prev_neuron_dims[2], m_prev_neuron_dims[1],
                                 m_prev_neuron_dims[0],
                                 this->m_model->get_max_mini_batch_size()};

    MPIPrintStreamDebug()
        << "input tensor shape: " << input_tensor_shape
        << ", desc: " << dc::util::tostring(this->m_prev_activations_cudnn_desc)
        << "\n";
    
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

    // View must be set every time fp is called, but it's also
    // necessary to call here as we need its memory property to be set
    // for setting up CUDNN tensor descriptors
    assert0(dc::tensor::View(
        m_prev_activations_e,
        m_prev_activations_d[0].get_locked_data(0)));
#if 0    
    MPIPrintStreamDebug() << "prev_activations_e: " <<
        m_prev_activations_e
                          << ", ptr: " << m_prev_activations_e.get_data()
                          << "\n";
#endif    

    // 1D decomposition at the H dimension
    Array4 spatial_decomposition = {1, m_comm->get_procs_per_model(), 1, 1};
    Array4 overlap = {m_pads[1], m_pads[0], 0, 0};
    Array4 spatial_block_size = {m_strides[1], m_strides[0], 1, 1};
    Array4 spatial_local_size = {0, 0, 0, 0};

    m_prev_activations_t = TensorDev(input_tensor_shape, loc,
                                     Dist(spatial_decomposition, overlap),
                                     spatial_local_size, spatial_block_size);
    m_prev_activations_t.allocate();
    m_prev_activations_t.zero();
    MPIPrintStreamDebug() << "prev_activations_t: " <<
        m_prev_activations_t << ", mem: " <<
        m_prev_activations_t.get_data() << "\n";

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
                                       Dist(spatial_decomposition, overlap));
    m_prev_error_signals_t.allocate();
    m_prev_error_signals_t.zero();
    m_prev_error_signals_e = ConstTensorDev(output_tensor_shape, loc,
                                            Dist(sample_decomposition),
                                            output_local_shape,
                                            division_block_size);
    
    assert0(dc::tensor::View(
        m_prev_error_signals_e,
        m_prev_error_signals_d[0].get_locked_data(0)));

    MPIPrintStreamDebug() << "prev_error_signals_e: " <<
        m_prev_error_signals_e << "\n";

    // error_signals
    m_error_signals_e = TensorDev(input_tensor_shape, loc,
                                  Dist(sample_decomposition),
                                  input_local_shape, division_block_size);
    m_error_signals_t = TensorDev(input_tensor_shape, loc,
                                  Dist(spatial_decomposition),
                                  spatial_local_size, spatial_block_size);
    m_error_signals_t.allocate();


    Array4 kernel_shape = {m_kernel_dims[3], m_kernel_dims[2],
                           m_kernel_dims[1], m_kernel_dims[0]};
    
    m_kernel_t = TensorDev(kernel_shape, loc, Dist());
    assert0(dc::tensor::View(
        m_kernel_t, m_weights[0]->get_values_gpu()[0]));
    m_kernel_gradient_e = TensorDev(kernel_shape, loc, Dist());
    assert0(dc::tensor::View(
        m_kernel_gradient_e, m_kernel_gradient_d.get_data(0)));
    
    m_conv = new dc::Convolution<dc::cudnn::BackendCUDNN>(
        *this->m_cudnn->get_distconv_backend());

    m_conv->setup(m_prev_activations_t,
                  m_kernel_t, m_activations_t,
                  m_error_signals_t, m_kernel_gradient_e,
                  m_prev_error_signals_t,
                  m_pads[0], m_pads[1],
                  m_strides[0], m_strides[1],
                  m_fwd_algo, m_bwd_data_algo,
                  m_bwd_filter_algo);

    // Bias tensor. Shared by all procs
    MPIPrintStreamDebug()
        << "Bias desc: "
        << dc::util::tostring(m_bias_cudnn_desc)
        << ", bias factor: " << m_bias_scaling_factor
        << "\n";
    if (m_bias_scaling_factor != DataType(0)) {
      Array4 bias_shape = {1, 1, this->m_neuron_dims[0], 1};
      m_bias_e = TensorDev(bias_shape, loc, Dist());
      assert0(dc::tensor::View(m_bias_e, m_weights[1]->get_values_gpu()[0]));
      MPIPrintStreamDebug()
          << "Bias tensor: " << m_bias_e << "\n";
      m_conv->setup_bias(m_bias_e);

      // Bias backprop
      optimizer* bias_optimizer = m_weights[1]->get_optimizer();      
      if (bias_optimizer != nullptr) {
        m_bias_gradient_e = TensorDev(bias_shape, loc, Dist());
        assert0(dc::tensor::View(m_bias_gradient_e,
                                 m_bias_gradient_d.get_data(0)));
        m_conv->setup_bias_gradient(m_bias_gradient_e);
      }
    }
        
        
#endif
  }
  
  void apply_convolution_distconv() {
#ifndef LBANN_HAS_DISTCONV
    throw lbann_exception(
        std::string {} + __FILE__ + " " + std::to_string(__LINE__) + " :: " +
        "Layer: DISTCONV not detected");
#else
    MPIPrintStreamDebug() << "Forward convolution\n";
    
    assert0(dc::tensor::View(
        m_prev_activations_e,
        m_prev_activations_d[0].get_locked_data(0)));
    MPIPrintStreamDebug() << "prev_activations_e: " <<
        m_prev_activations_e
                          << ", ptr: " << m_prev_activations_e.get_data()
                          << "\n";
    
    MPIPrintStreamDebug() << "Copying from data parallel to model parallel\n";

    dump_tensor(m_prev_activations_e, "prev_activations_original.txt");
    
    assert0(dc::tensor::Copy(
        m_prev_activations_t, m_prev_activations_e));

    dump_tensor(m_prev_activations_t, "prev_activations_spatial.txt");    

    assert0(dc::tensor::View(
        m_kernel_t, m_weights[0]->get_values_gpu()[0]));

    // there may only be a smaller number of samples for the last
    // mini-batch iteration
    m_conv->set_num_samples(this->m_model->get_current_mini_batch_size());
    m_conv->forward(1.0, m_prev_activations_t, m_kernel_t, 0.0, m_activations_t);

    dump_tensor(m_activations_t, "activations_spatial.txt");        

    assert0(dc::tensor::View(
        m_activations_e, m_activations_d[0].get_data(0)));

    dump_tensor(m_activations_e, "activations_original.txt");        
#endif
  }

  void apply_bias_distconv() {
#ifndef LBANN_HAS_DISTCONV
    throw lbann_exception(
        std::string {} + __FILE__ + " " + std::to_string(__LINE__) + " :: " +
        "Layer: DISTCONV not detected");
#else
    if (m_bias_scaling_factor == DataType(0)) return;

    MPIPrintStreamDebug() << "Applying bias\n";
    
    assert0(dc::tensor::View(
        m_bias_e, m_weights[1]->get_values_gpu()[0]));
    m_conv->apply_bias(m_bias_scaling_factor, m_bias_e,
                       1, m_activations_t);
#endif
  }

  void apply_transposed_convolution_distconv() {
#ifndef LBANN_HAS_DISTCONV
    throw lbann_exception(
        std::string {} + __FILE__ + " " + std::to_string(__LINE__) + " :: " +
        "Layer: DISTCONV not detected");
#else
    MPIPrintStreamDebug() << "Backward convolution\n";

    // input: m_prev_error_signals_d[0]
    // kernel: m_weights[0]->get_values_gpu()
    // output: m_error_signals_d[0]

    // Setup views
    assert0(dc::tensor::View(
        m_error_signals_e, m_error_signals_d[0].get_data(0)));
    assert0(dc::tensor::View(
        m_kernel_t, m_weights[0]->get_values_gpu()[0]));
    assert0(dc::tensor::View(
        m_prev_error_signals_e,
        m_prev_error_signals_d[0].get_locked_data(0)));
    
    // Copy to sample distribution
    if (!m_prev_error_signals_redistributed) {
      assert0(dc::tensor::Copy(
          m_prev_error_signals_t, m_prev_error_signals_e));
      m_prev_error_signals_redistributed = true;
    }

    dump_tensor(m_prev_error_signals_e,
                "prev_error_signals_original.txt");
    dump_tensor(m_prev_error_signals_t,
                "prev_error_signals_spatial.txt");

    assert0(dc::tensor::Copy(
        m_error_signals_t, m_error_signals_e));

    MPIPrintStreamDebug() << "Calling backward_data\n";
    m_conv->backward_data(1.0, m_kernel_t, m_prev_error_signals_t,
                          1.0, m_error_signals_t);

    dump_tensor(m_error_signals_t,
                "error_signals_spatial.txt");

    assert0(distconv::tensor::Copy(
        m_error_signals_e, m_error_signals_t));
    
#endif    
  }

  void compute_gradients_distconv() {
#ifndef LBANN_HAS_DISTCONV
    throw lbann_exception(
        std::string {} + __FILE__ + " " + std::to_string(__LINE__) + " :: " +
        "Layer: DISTCONV not detected");
#else
    MPIPrintStreamDebug() << "Compute gradients\n";

    assert0(dc::tensor::View(
        m_prev_error_signals_e,
        m_prev_error_signals_d[0].get_locked_data(0)));

    const int effective_mini_batch_size =
        this->m_model->get_effective_mini_batch_size();    

    optimizer* bias_optimizer = m_weights[1]->get_optimizer();
    if (bias_optimizer != nullptr && m_bias_scaling_factor != DataType(0)) {
      MPIPrintStreamDebug() << "Compute bias gradients\n";      
      // Copy to sample distribution
      assert0(dc::tensor::Copy(
          m_prev_error_signals_t, m_prev_error_signals_e));
      m_prev_error_signals_redistributed = true;
      assert0(dc::tensor::View(m_bias_gradient_e,
                               m_bias_gradient_d.get_data(0)));
      m_conv->backward_bias(1.0, m_prev_error_signals_t,
                            0.0, m_bias_gradient_e, false);
      const DataType bias_scale = m_bias_scaling_factor / effective_mini_batch_size;
      bias_optimizer->add_to_gradient_staging(m_bias_gradient_d,
                                              bias_scale);
      
    }

    optimizer* kernel_optimizer = m_weights[0]->get_optimizer();
    if (kernel_optimizer == nullptr) return;

    MPIPrintStreamDebug() << "Compute kernel gradients\n";          

    assert0(dc::tensor::View(
        m_kernel_gradient_e, m_kernel_gradient_d.get_data(0)));
    

#ifdef DEBUG_COMPUTE_GRADIENTS
    dump_tensor(m_kernel_gradient_e,
                "kernel_gradients_original.txt");
#endif

    // Copy to sample distribution
    if (!m_prev_error_signals_redistributed) {
      assert0(dc::tensor::Copy(
          m_prev_error_signals_t, m_prev_error_signals_e));
      m_prev_error_signals_redistributed = true;
    }

#ifdef DEBUG_COMPUTE_GRADIENTS
    Array4 kernel_shape = {m_kernel_dims[3], m_kernel_dims[2],
                           m_kernel_dims[1], m_kernel_dims[0]};
    LocaleMPI loc(m_comm->get_model_comm().comm);    
    TensorDev kg = TensorDev(kernel_shape, loc, Dist());
    kg.allocate();
    m_conv->backward_filter(1.0, m_prev_activations_t,
                            m_prev_error_signals_t, 0,
                            kg,
                            true);
    dump_tensor(kg, "kernel_gradients_spatial.txt");
#else
    m_conv->backward_filter(1.0, m_prev_activations_t,
                            m_prev_error_signals_t, 0,
                            m_kernel_gradient_e,
                            false);

    // Add gradient contribution
    const DataType kernel_scale = DataType(1) / effective_mini_batch_size;
    kernel_optimizer->add_to_gradient_staging(m_kernel_gradient_d,
                                              kernel_scale);
#endif    

#endif    
  }

  template <typename Tensor>
  void dump_tensor(const Tensor &t, const std::string &path) {
    if (m_dump_tensors) {
      dc::dump_tensor(t, path);
    }
  }

#ifdef LBANN_HAS_DISTCONV
  bool m_distconv_enabled = false;
  dc::Convolution<dc::cudnn::BackendCUDNN> *m_conv;
  /** Previous activation tensor */
  // Created once, view initialized at fp_setup_data
  TensorDev m_prev_activations_t;
  /** View to Elemental matrix of previous activations */
  // Created once, copied from m_prev_activations_t at fp_setup_data
  ConstTensorDev m_prev_activations_e;
  /** Activation tensor */
  // Created once, copied back to m_activations_e after fp_compute
  TensorDev m_activations_t;
  /** Elemental-format activation matrix */  
  TensorDev m_activations_e;
  /** Previous error signal tensor */
  TensorDev m_prev_error_signals_t;
  /** View to Elemental matrix */
  ConstTensorDev m_prev_error_signals_e;
  /** Error signal tensor */
  TensorDev m_error_signals_t;
  /** Elemental-format matrix */
  TensorDev m_error_signals_e;
  TensorDev m_kernel_t;
  TensorDev m_kernel_gradient_e;
  // Bias
  TensorDev m_bias_e;
  TensorDev m_bias_gradient_e;
  std::string m_fwd_algo = "DEFAULT";
  std::string m_bwd_data_algo = "DEFAULT";
  std::string m_bwd_filter_algo = "DEFAULT";

  bool m_prev_error_signals_redistributed = false;

  //bool m_dump_tensors = true;
  bool m_dump_tensors = false;  
#endif // LBANN_HAS_DISTCONV
  

};

} // namespace lbann

#endif // LBANN_LAYER_CONVOLUTION_HPP_INCLUDED

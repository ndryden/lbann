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
//
// lbann_data_reader_nci_regression .hpp .cpp - generic_data_reader class for National Cancer Institute (NCI) dataset
////////////////////////////////////////////////////////////////////////////////

#include "lbann/data_readers/data_reader_nci_regression.hpp"
#include <cstdio>
#include <string>

namespace lbann {

data_reader_nci_regression::data_reader_nci_regression(int batchSize, bool shuffle)
  : generic_data_reader(batchSize, shuffle) {
  m_num_samples = 0;
  //m_num_samples = -1;
  m_num_features = 0;
  m_num_responses = 1;
}

int data_reader_nci_regression::fetch_data(Mat& X) {
  if(!generic_data_reader::position_valid()) {
    return 0;
  }

  int current_batch_size = getm_batch_size();
  std::ifstream ifs(m_infile.c_str());
  if (!ifs) {
    std::cout << "\n In load: can't open file : " << m_infile;
    exit(1);
  }

  std::string line;
  int n = 0;
  for (n = m_current_pos; n < m_current_pos + current_batch_size; ++n) {
    if (n >= (int)m_shuffled_indices.size()) {
      break;
    }

    int k = n - m_current_pos;
    int index = m_shuffled_indices[n];

    std::getline(ifs.seekg(m_index_map[index]),line);
    std::istringstream lstream(line);
    std::string field;
    int col = 0, f=0;

    while(std::getline(lstream, field, ' ')) {
      col++;
      if (col == 3) {
        if (field.empty()) {
          break;
        } else {
          m_responses[index] = static_cast<DataType>(stof(field));
        }
      } else if (col > 5) {
        if(field.empty()) {
          X.Set(f, k, static_cast<DataType>(0));
        } else {
          X.Set(f,k,stof(field));
        }
        f++;
      }//end if col > 5
    }// end while loop
  } // end for loop (batch)
  ifs.close();
  return (n - m_current_pos);
}

int data_reader_nci_regression::fetch_response(Mat& Y) {
  if(!generic_data_reader::position_valid()) {
    return 0;
  }
  int current_batch_size = getm_batch_size();
  int n = 0;
  for (n = m_current_pos; n < m_current_pos + current_batch_size; ++n) {
    if (n >= (int)m_shuffled_indices.size()) {
      break;
    }

    int k = n - m_current_pos;
    int index = m_shuffled_indices[n];
    DataType sample_response = m_responses[index];

    Y.Set(0, k, sample_response);
  }
  return (n - m_current_pos);
}

/*Space separated columns are as follows (in order):
1) Drug Name
2) Cell line name
3) Drug response measurement
4) binary response label (derived from column 3 value)
5) ternary response label (derived from column 3 value and recommend we ignore for now)
6+) features*/

void data_reader_nci_regression::load() {
  std::string infile = get_data_filename();
  std::ifstream ifs(infile.c_str());
  if (!ifs) {
    throw lbann_exception(
      std::string{} + __FILE__ + " " + std::to_string(__LINE__) +
      " :: data_reader_nci_regression::load(): can't open file: " + infile);
  }
  m_infile = infile;
  std::string line;
  int i;

  m_index_map.push_back(ifs.tellg());
#if 1
  std::getline (ifs, line); // to skip the header
  m_index_map.pop_back();
  m_index_map.push_back(ifs.tellg());
#endif

  while(std::getline (ifs, line) ) {
#if 0 // enable to skip empty or comment lines
    static const std::string whitespaces(" \t\f\v\n\r");
    std::size_t pos = line.find_first_not_of(whitespaces);
    if ((pos == std::string::npos) || (line[pos] == '#')) {
      m_index_map.pop_back();
      m_index_map.push_back(ifs.tellg());
      continue;
    }
#endif

    m_index_map.push_back(ifs.tellg()); // The beginning of the next data to read
    m_num_samples++;

    std::string field;
    std::istringstream lstream(line);
    i=0;
    m_num_features = 0;
    while(std::getline(lstream, field, ' ')) {
      i++;
      if (i > 5) {
        m_num_features++;
      }
    }
  }
  ifs.close();
  m_index_map.pop_back();
  m_index_map.shrink_to_fit();
  m_responses.resize(m_num_samples);
  // reset indices
  m_shuffled_indices.clear();
  m_shuffled_indices.resize(m_num_samples);
  for (size_t n = 0; n < m_shuffled_indices.size(); ++n) {
    m_shuffled_indices[n] = n;
  }

  select_subset_of_data();
}

}  // namespace lbann

from lbann_onnx.l2o.layers import LbannLayerParser

class LbannLayerParser_batch_normalization(LbannLayerParser):
    def parse(self):
        params = self.l.batch_normalization
        return {"op": "BatchNormalization",
                "paramCount": 4,
                "attrs": {"epsilon":  params.epsilon,
                          "momentum": params.decay,
                          "spatial":  1}}

class LbannLayerParser_local_response_normalization(LbannLayerParser):
    def parse(self):
        params = self.l.local_response_normalization
        return {"op": "LRN",
                "attrs": {"alpha": params.lrn_alpha,
                          "beta":  params.lrn_beta,
                          "bias":  params.lrn_k,
                          "size":  params.window_width}}

class LbannLayerParser_dropout(LbannLayerParser):
    def parse(self):
        return {"op": "Dropout",
                "attrs": {"ratio": 1-self.l.dropout.keep_prob}}

# coding=utf-8
# Copyright cgmhaicenter and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" exBERT model configuration """

from ...configuration_utils import PretrainedConfig
from ...utils import logging

logger = logging.get_logger(__name__)

EXBERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "exbert": "https://huggingface.co/exbert/resolve/main/config.json",
    # See all exBERT models at https://huggingface.co/models?filter=exbert
}


class exBertConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a :class:`~transformers.exBertModel`.
    It is used to instantiate an exBERT model according to the specified arguments, defining the model
    architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of
    the exBERT `exbert <https://huggingface.co/exbert>`__ architecture.

    Configuration objects inherit from  :class:`~transformers.PretrainedConfig` and can be used
    to control the model outputs. Read the documentation from  :class:`~transformers.PretrainedConfig`
    for more information.


    Args:
        vocab_size (:obj:`int`, `optional`, defaults to 30000):
            Vocabulary size of the exBERT model. Defines the number of different tokens that can be represented by the
            :obj:`inputs_ids` passed when calling :class:`~transformers.exBertModel` or
            :class:`~transformers.TFexBertModel`.
        ex_vocab_size (:obj:`int`, `optional`, defaults to 10000):
            Vocabulary size of the extended module of exBERT model. Defines the number of different tokens that can be represented by the
            :obj:`inputs_ids` passed when calling :class:`~transformers.exBertModel` or
            :class:`~transformers.TFexBertModel`.
        hidden_size (:obj:`int`, `optional`, defaults to 768):
            Dimension of the encoder layers and the pooler layer.
        num_hidden_layers (:obj:`int`, `optional`, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        ex_num_hidden_layers (:obj:`int`, `optional`, defaults to 12):
            Number of hidden layers in the extended Transformer encoder.
        num_attention_heads (:obj:`int`, `optional`, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        ex_num_attention_heads (:obj:`int`, `optional`, defaults to 12):
            Number of attention heads for each attention layer in the extended Transformer encoder.
        intermediate_size (:obj:`int`, `optional`, defaults to 3072):
            Dimension of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        ex_intermediate_size (:obj:`int`, `optional`, defaults to 3072):
            Dimension of the "intermediate" (i.e., feed-forward) layer in the extended Transformer encoder.
        hidden_act (:obj:`str` or :obj:`function`, `optional`, defaults to :obj:`"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler.
            If string, :obj:`"gelu"`, :obj:`"relu"`, :obj:`"selu"` and :obj:`"gelu_new"` are supported.
        hidden_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):
            The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (:obj:`int`, `optional`, defaults to 300):
            The maximum sequence length that this model might ever be used with.
            Typically set this to something large just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (:obj:`int`, `optional`, defaults to 2):
            The vocabulary size of the :obj:`token_type_ids` passed when calling :class:`~transformers.exBertModel` or
            :class:`~transformers.TFexBertModel`.
        initializer_range (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (:obj:`float`, `optional`, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        use_cache (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if ``config.is_decoder=True``.
        gradient_checkpointing (:obj:`bool`, `optional`, defaults to :obj:`False`):
            If :obj:`True`, use gradient checkpointing to save memory at the expense of slower backward pass.
        base_model (:obj:`str`, defaults to :obj:`"beomi/kcbert-base"`):
            Base model to be frozen or trained with exBERT extended module.

        Example::

        >>> from transformers import exBertModel, exBertConfig

        >>> # Initializing a exBERT exbert style configuration
        >>> configuration = exBertConfig()

        >>> # Initializing a model from the exbert style configuration
        >>> model = exBertModel(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
    """
    model_type = "exbert"

    def __init__(
        self,
        vocab_size=30000,
        ex_vocab_size=20135,
        hidden_size=768,
        ex_hidden_size=252,
        num_hidden_layers=12,
        ex_num_hidden_layers=12,
        num_attention_heads=12,
        ex_num_attention_heads=12,
        intermediate_size=3072,
        ex_intermediate_size=1024,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=300,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        use_cache=True,
        is_encoder_decoder=False,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        base_model="beomi/kcbert-base",
        freeze_base_model=True,
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.ex_vocab_size = ex_vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.ex_hidden_size = ex_hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.ex_num_hidden_layers = ex_num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.ex_num_attention_heads = ex_num_attention_heads
        self.intermediate_size = intermediate_size
        self.ex_intermediate_size = ex_intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.type_vocab_size = type_vocab_size
        self.layer_norm_eps = layer_norm_eps
        self.use_cache = use_cache
        self.base_model = base_model
        self.freeze_base_model = freeze_base_model

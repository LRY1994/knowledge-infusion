import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import T5ForConditionalGeneration, BartForConditionalGeneration
from transformers.modeling_bart import shift_tokens_right

class MyBart(BartForConditionalGeneration):
    def forward(self, input_ids, attention_mask=None, encoder_outputs=None,
            decoder_input_ids=None, decoder_attention_mask=None, decoder_cached_states=None,
            use_cache=False, is_training=False):

        if is_training:
            _decoder_input_ids = shift_tokens_right(decoder_input_ids, self.config.pad_token_id)
        else:
            _decoder_input_ids = decoder_input_ids
        
        
       
        # print(input_ids.shape) #    
        # print(attention_mask.shape)
        # print(encoder_outputs)#none
        # print(decoder_input_ids.shape)
        # print(decoder_attention_mask.shape) 
        # print(decoder_cached_states)
        # print(use_cache)#False



        outputs = self.model(
            input_ids,#torch.Size([4, 32]) batchsize,max_input_length
            attention_mask=attention_mask,#torch.Size([4, 32]) batchsize,max_input_length
            encoder_outputs=encoder_outputs,
            decoder_input_ids=_decoder_input_ids,#torch.Size([4, 36]) batchsize, 默认output_length
            decoder_attention_mask=decoder_attention_mask,#torch.Size([4, 36]) batchsize, 默认output_length
            decoder_cached_states=decoder_cached_states,
            use_cache=use_cache,
        )

           
        # print(outputs[0].shape)#torch.Size([4, 36, 1024])
        # print(self.model.shared.weight.shape)#torch.Size([50265, 1024])
        # print(self.final_logits_bias.shape)#torch.Size([1, 50265])
    

        lm_logits = F.linear(outputs[0], self.model.shared.weight, bias=self.final_logits_bias)
        if is_training:
            loss_fct = nn.CrossEntropyLoss(reduction="sum", ignore_index=self.config.pad_token_id)
            loss = loss_fct(lm_logits.view(-1, self.config.vocab_size),
                              decoder_input_ids.view(-1))
            return loss
        return (lm_logits, ) + outputs[1:]


import torch
import torch.nn as nn
from torch.utils.data import Dataset

# Bộ dữ liệu song ngữ
class BilingualDataset(Dataset):

    def __init__(self,
                 ds,    # Dữ liệu gốc
                 tokenizer_src, # Tokenizer nguồn
                 tokenizer_tgt, # Tokenizer đích
                 src_lang,  # Ngôn ngữ nguồn
                 tgt_lang,  # Ngôn ngữ đích
                 seq_len    # Độ dài tối đa của một câu
                ):
        super().__init__()
        
        self.seq_len = seq_len
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        
        # Ba biến self.sos_token, self.eos_token, và self.pad_token được khởi tạo bằng cách sử dụng tokenizer của ngôn ngữ đích (tokenizer_tgt).
        # Các biến này chứa các mã số nguyên tương ứng với các token đặc biệt
        # Được sử dụng trong quá trình xử lý dữ liệu và huấn luyện mô hình dịch máy.

        
        # Token này được sử dụng để chỉ định vị trí bắt đầu của một câu trong mô hình dịch máy.
        # Nó được thêm vào đầu câu đích để cho biết mô hình bắt đầu dịch từ vị trí này.
        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        
        # Token này được sử dụng để chỉ định vị trí kết thúc của một câu trong mô hình dịch máy.
        # Nó được thêm vào cuối câu đích để cho biết mô hình dừng quá trình dịch tại vị trí này.
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        
        # Token này được sử dụng để đánh dấu các vị trí trong câu đích hoặc câu nguồn không có từ tương ứng.
        # Khi các câu có độ dài khác nhau, chúng cần được điền đầy đủ thành các chuỗi có cùng độ dài.
        # Token "[PAD]" được sử dụng để thêm vào cuối câu để đạt được độ dài mong muốn.
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)

    # Hàm trả về độ dài của dữ liệu
    def __len__(self):
        return len(self.ds)

    # Hàm lấy ra 1 iteam
    def __getitem__(self,
                    idx # Vị trí một phần tử dữ liệu trong dataset
                    ):
        
        # Lấy ra phần tử dữ liệu
        src_target_pair = self.ds[idx]
        
        # Lấy ra câu nguồn trong phần tử dữ liệu
        src_text = src_target_pair['translation'][self.src_lang]
        
        # Lấy ra câu đích trong phần tử dữ liệu
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        # enc_input_tokens là kết quả của việc mã hóa câu nguồn (src_text) bằng tokenizer của ngôn ngữ nguồn (tokenizer_src).
        # Mã hóa được thực hiện bằng cách gọi phương thức encode trên tokenizer_src
        # và truy cập thuộc tính ids để lấy danh sách các mã số nguyên tương ứng với các token trong câu nguồn.
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        
        # Tương tự, dec_input_tokens là kết quả của việc mã hóa câu đích (tgt_text) bằng tokenizer của ngôn ngữ đích (tokenizer_tgt).
        # Mã hóa cũng được thực hiện bằng cách gọi phương thức encode trên tokenizer_tgt và
        # truy cập thuộc tính ids để lấy danh sách các mã số nguyên tương ứng với các token trong câu đích.
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        
        # enc_num_padding_tokens tính toán số lượng token "[PAD]" cần thêm vào câu nguồn (enc_input_tokens)
        # để đảm bảo độ dài của câu nguồn bằng self.seq_len (độ dài câu đã được quy định trước) sau khi thêm token "[SOS]" và "[EOS]".
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
        
        # dec_num_padding_tokens tính toán số lượng token "[PAD]" cần thêm vào câu đích (dec_input_tokens)
        # để đảm bảo độ dài của câu đích bằng self.seq_len sau khi thêm token "[SOS]".
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        
        # Nếu số lượng token "[PAD]" cần thêm là âm (nhỏ hơn 0), điều đó có nghĩa là câu quá dài và không thể chứa trong self.seq_len.
        # Trong trường hợp này, một lỗi ValueError được ném ra.
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")

        # encoder_input là đầu vào của bộ mã hóa và được tạo bằng cách ghép các tensor theo chiều 0.
        # Nó bao gồm token "[SOS]", danh sách mã số nguyên enc_input_tokens,
        # token "[EOS]" và danh sách mã số nguyên "[PAD]" để đạt đủ độ dài self.seq_len.
        # encoder_input: ([SOS], [encoding input], [EOS], [PAD])
        encoder_input = torch.cat(
            [
                self.sos_token, # Token bắt đầu
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token, # Token kết thúc
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64), # Số lượng token padding
            ],
            dim=0,
        )

        # decoder_input là đầu vào của bộ giải mã và được tạo tương tự như encoder_input,
        # nhưng chỉ bao gồm token "[SOS]", danh sách mã số nguyên dec_input_tokens và danh sách mã số nguyên "[PAD]".
        # decoder_input: ([SOS], [decoding input], [PAD])
        decoder_input = torch.cat(
            [
                self.sos_token, # Token bắt đầu
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64), # Số lượng token padding
            ],
            dim=0,
        )
        # Quá trình này đảm bảo rằng các câu nguồn và câu đích có cùng độ dài self.seq_len,
        # và đã được bổ sung đầy đủ các token đặc biệt để dùng trong mô hình dịch máy.
        
        
        

        # Trong đoạn mã trên, biến label được tạo để lưu trữ đầu ra dự kiến (labels) của mô hình dịch máy.
        # label được tạo bằng cách sử dụng các tensor được ghép lại theo chiều 0.
        # Đầu tiên, danh sách mã số nguyên dec_input_tokens được chuyển đổi thành một tensor và được thêm vào label.
        # Tiếp theo, token self.eos_token (token kết thúc câu) được thêm vào label.
        # Cuối cùng, danh sách mã số nguyên "[PAD]" với độ dài dec_num_padding_tokens được chuyển đổi thành một tensor và được thêm vào label.
        # Kết quả là label chứa một chuỗi các mã số nguyên tương ứng với câu đích đã được mở rộng để có độ dài self.seq_len và có thêm token kết thúc câu ("[EOS]") và token đệm ("[PAD]") nếu cần thiết.
        # label: ([decoding input], [EOS], [PAD])
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Double check the size of the tensors to make sure they are all seq_len long
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,  # (seq_len)
            "decoder_input": decoder_input,  # (seq_len)
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1, seq_len) & (1, seq_len, seq_len),
            "label": label,  # (seq_len)
            "src_text": src_text,
            "tgt_text": tgt_text,
        }
    
def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0

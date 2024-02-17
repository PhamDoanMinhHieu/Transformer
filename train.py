from model import build_transformer
from dataset import BilingualDataset, causal_mask
from config import get_config, get_weights_file_path, latest_weights_file_path

import pip  as datasets
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import LambdaLR

import warnings
import wandb
from tqdm import tqdm
import os
from pathlib import Path

# Huggingface datasets and tokenizers
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

import torchmetrics

from torch.utils.tensorboard import SummaryWriter


# Hàm lấy các câu trong dữ liệu với ngôn ngữ xác định
def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][str(lang)]

# Hàm lấy dữ liệu
def get_ds(config # Cấu hình
           ):
    
    # Hàm load_dataset để tải tập dữ liệu từ nguồn dữ liệu được chỉnh định trong config
    # Tập dữ liệu này có phần chia train nên chúng ta sẽ tự chia lại thành tập huấn luyện và tập validation
    # ds_ras là dữ liệu thô
    ds_raw = dataset = load_dataset("wmt16", "cs-en", split="train")

    # Xây dựng tokenizer cho encoder và decoder
    tokenizer_src = get_or_build_tokenizer(config, # cấu hình
                                           ds_raw, # dữ liệu gốc
                                           config['lang_src']) # ngôn ngữ nguồn
    
    tokenizer_tgt = get_or_build_tokenizer(config, # file cấu hình
                                           ds_raw, # dữ liệu gốc
                                           config['lang_tgt']) # ngôn ngữ đích

    # Chia tập dữ liệu thành 90% cho training, 10% cho validation bằng thư viện random_split
    train_ds_size = int(0.9 * len(ds_raw)) # Lấy size train
    val_ds_size = len(ds_raw) - train_ds_size   # Lấy size test
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    # Thiết lập train dataset
    train_ds = BilingualDataset(train_ds_raw, # dữ liệu train (90% dữ liệu gốc)
                                tokenizer_src,  # input tokenizer
                                tokenizer_tgt,  # output tokenizer
                                config['lang_src'], # cấu hình ngôn ngữ nguồn
                                config['lang_tgt'], # cấu hình ngôn ngữ đích
                                config['seq_len'])  # cấu hình độ dài của câu
    
    # Thiết lập validation dataset
    val_ds = BilingualDataset(val_ds_raw, # dữ liệu validation (10% dữ liệu gốc)
                              tokenizer_src, # input tokenizer
                              tokenizer_tgt, # output tokenizer
                              config['lang_src'],  # cấu hình ngôn ngữ nguồn
                              config['lang_tgt'],  # cấu hình ngôn ngữ đích
                              config['seq_len'])   # cấu hình độ dài của câu

    # Tìm chiều dài lớn nhất của câu nguồn và câu đích bằng cách mã hóa các câu và tính độ dài mã hóa
    max_len_src = 0
    max_len_tgt = 0

    # Duyệt qua các câu trong dữ liệu gốc
    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')
    
    # Khởi tạp các DataLoader cho tập train và valiatation
    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    
    # Tóm lại, hàm này thực hiện các bước như tải tập dữ liệu, xây dựng tokenizer, chia tập huấn luyện và tập validation,
    # tính độ dài lớn nhất của câu, và trả về các dataloader và tokenizer cần thiết cho việc huấn luyện và đánh giá mô hình dịch máy.
    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


# Xây dựng hoặc lấy tokenizer
def get_or_build_tokenizer(config, # cấu hình
                           ds,     # dữ liệu
                           lang):  # ngôn ngữ 
    
    # Đường dẫn tới file tokenizer dựa vào config và lang
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    
    # Kiểm tra xem file tokenizer có tồn tại
    if not Path.exists(tokenizer_path):
        
        # Khởi tạo đối tượng tokenizer, sử dụng [UNK] đại diện cho những từ không được biết đến
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        
        # Tokennizer sẽ tách các từ trong câu dựa trên khoảng trắng
        tokenizer.pre_tokenizer = Whitespace()
        
        # WordLevelTrainer được sử dụng để huấn luyện tokenizer
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        
        # Thực hiện quá trình train
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        
        # Lưu lại kết quả train vào đường dẫn
        tokenizer.save(str(tokenizer_path))
    else:
        # Tải lại tokenizer được lưu trữ từ file trước đó
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    
    return tokenizer

# Hàm xây dựng model
def get_model(config, # Cấu hình
              vocab_src_len, # Chiều dài từ điển nguồn
              vocab_tgt_len # Chiều dài từ điển đích
              ):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config["seq_len"], config['seq_len'], d_model=config['d_model'])
    return model

# Thực hiện giải mã bằng phương pháp Greedy Decoding.
def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    # Lấy chỉ số của token "[SOS]" và "[EOS]" từ tokenizer của đầu ra (target).
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # ính toán đầu ra của encoder bằng cách chạy đầu vào (source) và mặt nạ (source_mask) qua mô hình.
    # Đầu ra này được lưu trong biến encoder_output và sẽ được sử dụng lại cho mỗi bước giải mã.
    encoder_output = model.encode(source, source_mask)
    
    # Khởi tạo đầu vào của decoder bằng một tensor rỗng có kích thước (1, 1) và giá trị của sos_idx.
    # Đầu vào được chuyển sang thiết bị tính toán đã chọn (device).
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    
    # Vòng lặp while True được sử dụng để thực hiện giải mã các từ tiếp theo cho đến khi gặp ký hiệu kết thúc ([EOS]) hoặc đạt đến độ dài tối đa (max_len).
    while True:
        if decoder_input.size(1) == max_len:
            break

        # Xây dựng mặt nạ (decoder_mask) cho decoder dựa trên độ dài hiện tại của đầu vào của decoder (decoder_input).
        # Mặt nạ này sẽ được sử dụng để bỏ qua các vị trí không cần thiết trong quá trình tính toán.
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # Tính toán đầu ra của decoder bằng cách chạy đầu ra của encoder (encoder_output),
        # mặt nạ của encoder (source_mask), đầu vào của decoder (decoder_input) và mặt nạ của decoder (decoder_mask) qua mô hình.
        # Đầu ra này được lưu trong biến out.
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # Ánh xạ đầu ra của decoder qua projection layer để tính toán xác suất của từ tiếp theo.
        # out[:, -1] trích xuất đầu ra ứng với vị trí cuối cùng của mỗi chuỗi đầu ra trong batch.
        prob = model.project(out[:, -1])
        
        # Chọn từ có xác suất cao nhất làm từ tiếp theo. torch.max trả về giá trị lớn nhất và chỉ số của nó trong tensor prob.
        # Chỉ số của từ tiếp theo được lưu trong biến next_word.
        _, next_word = torch.max(prob, dim=1)
        
        # Thêm từ tiếp theo vào đầu vào của decoder bằng cách nối (torch.cat) từ tiếp theo với decoder_input.
        # Đầu vào của decoder được cập nhật để sử dụng từ tiếp theo cho bước giải mã tiếp theo.
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )

        # Nếu từ tiếp theo là ký hiệu kết thúc ([EOS]), thì dừng quá trình giải mã.
        if next_word == eos_idx:
            break
    # Trả về đầu vào của decoder đã giải mã, sau khi loại bỏ kích thước batch thêm vào trong quá trình giải mã.
    return decoder_input.squeeze(0)

# Hàm đánh giá mô hình
def run_validation(model, # Mô hình
                   validation_ds, # Dữ liệu đánh giá
                   tokenizer_src, # tokenizer nguồn
                   tokenizer_tgt, # tokenizer đích
                   max_len, # chiều dài tối đa của một câu
                   device, # thiết bị tính toán
                   print_msg,
                   global_step,
                   writer,
                   num_examples=2
                   ):
    
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    try:
        # get the console window width
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        # If we can't get the console width, use 80 as default
        console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device) # (b, seq_len)
            encoder_mask = batch["encoder_mask"].to(device) # (b, 1, 1, seq_len)

            # check that the batch size is 1
            assert encoder_input.size(
                0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)
            
            # Print the source, target and model output
            print_msg('-'*console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print_msg('-'*console_width)
                break
    
    if writer:
        # Evaluate the character error rate
        # Compute the char error rate 
        metric = torchmetrics.CharErrorRate()
        cer = metric(predicted, expected)
        writer.add_scalar('validation cer', cer, global_step)
        writer.flush()

        # Compute the word error rate
        metric = torchmetrics.WordErrorRate()
        wer = metric(predicted, expected)
        writer.add_scalar('validation wer', wer, global_step)
        writer.flush()

        # Compute the BLEU metric
        metric = torchmetrics.BLEUScore()
        bleu = metric(predicted, expected)
        writer.add_scalar('validation BLEU', bleu, global_step)
        writer.flush()





# Hàm huấn luyện model
def train_model(config):
    # kiểm tra và cấu hình thiết bị tính toán cho việc huấn luyện mô hình sử dụng PyTorch trong môi trường máy tính. #
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)
    
    if (device == 'cuda'):
        print(f"Device name: {torch.cuda.get_device_name(device.index)}")
        print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")
    elif (device == 'mps'):
        print(f"Device name: <mps>")
    else:
        print("NOTE: If you have a GPU, consider using it for training.")
        print("      On a Windows machine with NVidia GPU, check this video: https://www.youtube.com/watch?v=GMSjDTU8Zlc")
        print("      On a Mac machine, run: pip3 install --pre torch torchvision torchaudio torchtext --index-url https://download.pytorch.org/whl/nightly/cpu")
    device = torch.device(device)
    # ============================================================================================================== #

    # Chuẩn bị dữ liệu và xây dựng mô hình
    Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    # ===================================== # 
    
    
    # Cập nhập tham số và theo dõi tiến trình 
    # SummaryWriter được sử dụng để ghi lại thông tin và sự tiến triển của quá trình huấn luyện mô hình
    writer = SummaryWriter(config['experiment_name'])

    # Bộ tối ưu hóa Adam được sử dụng để cập nhật các tham số của mô hình trong quá trình huấn luyện.
    # Trong trường hợp này, các tham số cần được cập nhật là model.parameters(), tức là các tham số trong mô hình
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)
    # ====================================== #


    # Bắt đầu quá trình huấn luyện
    initial_epoch = 0 # Đây là biến dùng để theo dõi epoch hiện tại trong quá trình huấn luyện.
    global_step = 0 # Đây là biến dùng để theo dõi số lượng bước huấn luyện (steps) đã được thực hiện trong quá trình huấn luyện.
    preload = config['preload'] # Biến này xác định liệu có sử dụng mô hình được tải trước đó (preloaded) hay không.
    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    # Nếu tệp tin lưu trữ trọng số model không none
    if model_filename:
        print(f'Preloading model {model_filename}')
        # Tiến hành tải trọng số mô hình đó và tiếp tục từ epoch tiếp theo, bước đầu tiên và trạng thái tối ưu hóa đã lưu trữ.
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print('No model to preload, starting from scratch')

    # Hàm mất mát này được sử dụng để tính toán lỗi trong quá trình huấn luyện mô hình.
    # Tham số ignore_index được đặt bằng chỉ số của token '[PAD]' trong từ điển tokenizer tokenizer_src,
    # để bỏ qua việc tính toán lỗi cho các vị trí được gán nhãn là '[PAD]'.
    # Tham số label_smoothing được đặt bằng giá trị 0.1, để áp dụng kỹ thuật làm mịn nhãn (label smoothing) trong quá trình tính toán lỗi.
    # Cuối cùng, hàm mất mát được đặt trên thiết bị tính toán đã chọn (device).
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    # Duyệt qua từng epoch
    for epoch in range(initial_epoch, config['num_epochs']):
        # Giải phóng bộ nhớ GPU
        torch.cuda.empty_cache()
        
        # Thực hiện huân luyện model
        model.train()
        
        # tqdm là một công cụ tiến trình (progress bar) được sử dụng để hiển thị tiến độ của vòng lặp
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        
        # Duyệt qua các batch
        for batch in batch_iterator:

            # Lấy đầu vào của encoder từ batch hiện tại và chuyển nó sang thiết bị tính toán đã chọn (device).
            # encoder_input là một tensor có kích thước (B, seq_len), với B là kích thước batch và seq_len là độ dài của mỗi chuỗi đầu vào.
            encoder_input = batch['encoder_input'].to(device) # (B, seq_len)
            
            # Lấy đầu vào của decoder từ batch hiện tại và chuyển nó sang thiết bị tính toán đã chọn (device).
            # decoder_input là một tensor có kích thước (B, seq_len),
            decoder_input = batch['decoder_input'].to(device) # (B, seq_len)
            
            # Lấy mask (mặt nạ) của encoder từ batch hiện tại và chuyển nó sang thiết bị tính toán đã chọn (device).
            # encoder_mask là một tensor có kích thước (B, 1, 1, seq_len),
            # được sử dụng để bỏ qua các vị trí không cần thiết trong quá trình tính toán.
            encoder_mask = batch['encoder_mask'].to(device) # (B, 1, 1, seq_len)
            
            # Lấy mask (mặt nạ) của decoder từ batch hiện tại và chuyển nó sang thiết bị tính toán đã chọn (device).
            # decoder_mask cũng có kích thước (B, 1, seq_len, seq_len)
            # và được sử dụng để bỏ qua các vị trí không cần thiết trong quá trình tính toán.
            decoder_mask = batch['decoder_mask'].to(device) # (B, 1, seq_len, seq_len)

            
            # Chạy đầu vào của encoder và mask qua mô hình để tính toán đầu ra của encoder.
            # encoder_output là một tensor có kích thước (B, seq_len, d_model),
            # trong đó d_model là số chiều của vector biểu diễn (embedding) tại mỗi vị trí đầu vào.
            encoder_output = model.encode(encoder_input, encoder_mask) # (B, seq_len, d_model)
            
            # Chạy đầu ra của encoder, mask của encoder, đầu vào của decoder và mask của decoder qua mô hình để tính toán đầu ra của decoder.
            # decoder_output có kích thước (B, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (B, seq_len, d_model)
            
            # Chạy đầu ra của decoder qua mô hình projection layer để ánh xạ từ không gian vector biểu diễn về không gian từ vựng.
            # proj_output có kích thước (B, seq_len, vocab_size), trong đó vocab_size là kích thước của từ điển đầu ra.
            proj_output = model.project(decoder_output) # (B, seq_len, vocab_size)

            # Lấy nhãn (label) từ batch hiện tại và chuyển nó sang thiết bị tính toán đã chọn (device).
            # label cũng có kích thước (B, seq_len), tương tự như encoder_input và decoder_input.
            label = batch['label'].to(device) # (B, seq_len)

            # Tính toán hàm mất mát (loss) bằng cách so sánh đầu ra của projection layer (proj_output) với nhãn (label).
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            
            # Cập nhật tiến trình hiện tại của vòng lặp và hiển thị loss trong thanh tiến trình.
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # Hiển thị loss của batch hiện tại
            writer.add_scalar('train loss', loss.item(), global_step)
            
            # Đẩy dữ liệu logs vào TensorBoard để lưu trữ.
            writer.flush()

            # Tính toán gradient của loss theo các tham số trong mô hình để chuẩn bị cho việc cập nhật trọng số (backpropagation).
            loss.backward()

            # Cập nhật trọng số của mô hình dựa trên gradient tính toán được từ loss.backward().
            optimizer.step()
            
            # Đặt gradient của các tham số về zero để chuẩn bị cho lần tính toán gradient tiếp theo.
            optimizer.zero_grad(set_to_none=True)

            # Tăng biến global_step lên mỗi lần lặp để theo dõi số lần cập nhật trọng số đã được thực hiện.
            global_step += 1

        # Đánh giá chất lượng mô hình sau tất cả vòng lặp
        run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)

        # Lưu lại model
        # Đây là một hàm được sử dụng để tạo đường dẫn đến tệp tin lưu trữ trọng số của mô hình.
        # Đường dẫn này được tạo dựa trên config (cấu hình) và số epoch (epoch) hiện tại, được định dạng thành hai chữ số ({epoch:02d}).
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        
        # Hàm torch.save() được sử dụng để lưu trữ trạng thái của mô hình
        torch.save({
            'epoch': epoch, # Lưu trữ số epoch hiện tại vào tệp tin.
            'model_state_dict': model.state_dict(), # Lưu trữ trạng thái của mô hình (các trọng số và bias) vào tệp tin.
            'optimizer_state_dict': optimizer.state_dict(), # Lưu trữ trạng thái của trình tối ưu hóa (các trạng thái và siêu tham số) vào tệp tin.
            'global_step': global_step # Lưu trữ số bước toàn cục hiện tại vào tệp tin.
        }, model_filename)
        
        # Việc lưu trữ trạng thái của mô hình và trình tối ưu hóa cho phép chúng ta lưu lại tiến trình huấn luyện và khôi phục lại từ trạng thái đã lưu khi cần thiết.

if __name__ == '__main__':
    # warnings.filterwarnings("ignore")
    # config = get_config()
    # train_model(config)
    
    
    # Đoạn mã này được sử dụng để tắt cảnh báo (warnings) trong quá trình chạy mã.
    # Các cảnh báo thông thường sẽ không được hiển thị trong quá trình thực thi.
    warnings.filterwarnings("ignore")
    
    # để lấy cấu hình (config) cho quá trình huấn luyện mô hình.
    # Cấu hình này chứa các thông số cần thiết như số lượng epoch, tệp tin dữ liệu, kích thước batch, và các siêu tham số khác.
    config = get_config()
    
    # đặt số lượng epoch (vòng lặp huấn luyện) là 30.
    config['num_epochs'] = 30
    
    # chỉ định rằng không có mô hình đã được tải trước (preloaded model).
    config['preload'] = None

    # Khởi tạo một run mới trong Weights & Biases (W&B).
    # Weights & Biases là một công cụ giúp ghi lại và theo dõi các thí nghiệm và huấn luyện mô hình.
    # Hàm wandb.init() được sử dụng để khởi tạo một run mới với các thông số cấu hình được đặt trong tham số config, và dự án (project) được đặt thành "pytorch-transformer".
    # wandb.init(
    #     # set the wandb project where this run will be logged
    #     project="pytorch-transformer",
        
    #     # track hyperparameters and run metadata
    #     config=config
    # )
    
    # để bắt đầu quá trình huấn luyện mô hình
    train_model(config)
import os 
import regex as re
from collections import defaultdict,Counter
import json

def train_bpe(
    input_path:str|os.PathLike,  #输入文件路径
    vocab_size:int,
    special_tokens:list[str],
) -> tuple[dict[int,bytes],list[tuple[bytes,bytes]]]:
    
    """
    训练字节级 BPE (Byte-Pair Encoding) 分词器。
    
    该函数 BPE 算法的核心流程：
    1. 初始化词表为所有可能的字节 (0-255)。
    2.  读取输入语料，并根据特殊 Token 进行切分，确保特殊 Token 不参与统计。
    3. 使用 GPT-2 的预分词正则将语料库切分成单词，并统计每个单词的频率。
    4. 迭代进行“合并”操作，直到达到目标词表大小。
       - 合并策略：总是选择当前出现频率最高、且在字典序上最大的字节对。
    5. 使用倒排索引优化合并过程中的频率更新，确保速度。
    6. 将合并产生的 Token 加入词表，并最终加入特殊 Token。
    
    返回:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab: 训练好的词汇表，映射 Token ID -> Token 字节序列。
            merges: BPE 合并规则列表，按生成顺序排列。
    """
    # 1.初始化基础词表
    vocab={i:bytes([i]) for i in range(256)}
    # 计算需要合并的次数
    num_merges=vocab_size-256-len(special_tokens)
    
    # 2.读取数据
    with open(input_path,"r",encoding='utf-8') as f:
        text=f.read()
        
        
    #3.特殊token处理
    if special_tokens:
        #链接正则表达式
        special_regex="|".join(re.escape(t) for t in special_tokens)
        # 正则切分 采用捕获组
        parts=re.split(f"({special_regex})",text)
        # 筛选得到最终片段
        train_segments=[p for p in parts if p not in special_tokens]
    else:
        # 没有特殊token则使用整个文本
        train_segments=[text]
        
    
    # 4.预分词
    # 采用GPT-2官方的预分词正则表达式，将segments切分成”单词“
    gpt2_pat = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    
    # 预分词并统计所有的单词及其出现频率
    raw_counts=Counter()
    for segment in train_segments:
        words=gpt2_pat.findall(segment)
        for word in words:
            raw_counts[tuple(bytes([b]) for b in word.encode('utf-8'))]+=1
            
    # 构建高效的合并和索引列表
    words_list=[]
    counts_list=[]
    # 将raw_counts中的数据归纳整理
    for word_tuple,freq in raw_counts.items():
        words_list.append(list(word_tuple))
        counts_list.append(freq)
        
    stats=defaultdict(int)
    
    indices=defaultdict(set)
    
    for idx,word in enumerate(words_list):
        freq=counts_list[idx]
        for i in range(len(word)-1):
            pair=(word[i],word[i+1])
            stats[pair]+=freq
            indices[pair].add(idx)
            
    merges=[]
    
    # 开始合并
    for _ in range(num_merges):
        
        if not stats:
            break
        
        #寻找最佳pair
        best_pair=max(stats.items(),key=lambda x :(x[1],x[0]))[0]
        
        
        if stats[best_pair]<=0:
            break
        
        merges.append(best_pair)
        
        new_token=best_pair[0]+best_pair[1]
        
        # 更新受影响的单词
        relevant_indices=list(indices[best_pair])
        for idx in relevant_indices:
            word=words_list[idx]
            freq=counts_list[idx]
            
            i=0
            while i<len(word)-1:
                if word[i]==best_pair[0] and word[i+1]==best_pair[1]:
                    # 更新左邻居
                    if i > 0:
                        prev_pair=(word[i-1],word[i])
                        stats[prev_pair]-=freq
                        if stats[prev_pair]==0:
                            del stats[prev_pair]
                    
                    # 更新右邻居
                    if i < len(word)-2:
                        next_pair=(word[i+1],word[i+2])
                        stats[next_pair]-=freq
                        if stats[next_pair]==0:
                            del stats[next_pair]
                
                    # 修改单词结构
                    word[i]=new_token
                    del word[i+1]
                    
                    # 添加新邻居的频率和索引
                    if i > 0:
                        new_prev=(word[i-1],word[i])
                        stats[new_prev]+=freq
                        indices[new_prev].add(idx)
                        
                    if i < len(word)-1:
                        new_next=(word[i],word[i+1])
                        stats[new_next]+=freq
                        indices[new_next].add(idx)
                        
                else:
                    i+=1
            
                
        #清理合并后的best_pair
        if best_pair in stats: del stats[best_pair]
        if best_pair in indices:del indices[best_pair] 
                       
    
    # 构建最终vocab
    for pair in merges:
        new_id=len(vocab)
        vocab[new_id]=pair[0]+pair[1]
        
    for s_tok in special_tokens:
        s_bytes = s_tok.encode("utf-8")
        vocab[len(vocab)]=s_bytes
        
    return vocab,merges
    
    
def bytes_to_unicode():
    """
    创建一个映射，将 0-255 字节映射为一组可见的 Unicode 字符。
    这是 GPT-2 源码中的标准做法。
    """
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def save_tokenizer_files(vocab, merges, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # 初始化映射表
    byte_encoder = bytes_to_unicode()

    # 词表保存
    # 使用 byte_encoder 将 bytes 转换为可见字符串
    json_vocab = {
        k: "".join(byte_encoder[b] for b in v) 
        for k, v in vocab.items()
    }
    with open(os.path.join(out_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(json_vocab, f, indent=4)
    
    # 合并规则保存
    with open(os.path.join(out_dir, "merges.txt"), "w", encoding="utf-8") as f:
        for p1, p2 in merges:
            # 同样转换 p1 和 p2
            s1 = "".join(byte_encoder[b] for b in p1)
            s2 = "".join(byte_encoder[b] for b in p2)
            f.write(f"{s1} {s2}\n")
    
    
def main():
    input_path="data/TinyStoriesV2-GPT4-valid.txt"
    vocab_size=10000
    
    special_tokens=["<|endoftext|>"]
    output_dir="data/TinyStoriesV2-GPT4-valid"
    
    # 开始训练
    vocab,merges=train_bpe(input_path,vocab_size,special_tokens)
    # 保存结果
    save_tokenizer_files(vocab,merges,output_dir)
    
if __name__=="__main__":
    main()
    









import regex as re
from collections.abc import Iterable

"""
For special_tokens:
    推理/编码阶段 (Tokenizer.encode)
        在模型使用分词器将文本转为 ID 时，必须优先匹配特殊 Token。
    代码逻辑：
        正则匹配：构建一个包含所有特殊 Token 的正则表达式。
        优先级：先扫描文本，一旦发现特殊 Token，直接将其转为对应的 ID。
        普通处理：特殊 Token 之间的文本，再走正常的 GPT-2 预分词和 BPE 合并流程。
"""

class BPETokenizer:
    """
    字节级 BPE（Byte-Pair Encoding）分词器实现。
    
    该分词器将任意字符串编码为整数 ID 序列，并能将 ID 序列还原。
    它采用字节级处理，确保不会出现未知词（OOV）错误。
    """
    
    def __init__(self,vocab:dict[int,bytes],merges:list[tuple[bytes,bytes]],special_tokens:list[str]|None=None) -> None:
        """
        初始化分词器。
        
        参数:
            vocab: 词汇表，建立整数 ID 到 字节块(bytes) 的映射。
            merges: 合并规则列表。列表中的每一项是一个二元组 (bytes_a, bytes_b)，
                   表示在训练过程中 bytes_a 和 bytes_b 被合并的顺序。
            special_tokens: 特殊标记列表（如 <|endoftext|>），这些标记不会被 BPE 规则拆分。
        """
        
        # 1.建立双向映射，便于查表
        self.vocab=vocab 
        self.id_to_byte=vocab
        self.byte_to_id={v:k for k,v in vocab.items()}
        
        # 2.将合并规则转换为rank字典
        # 优先应用在训练阶段较早出现的合并规则
        self.merges={pair : i for i,pair in enumerate(merges)}
        
        self.special_tokens=special_tokens or []
        
        # 3.构建特殊token的正则表达式
        if self.special_tokens:
            sorted_special=sorted(self.special_tokens,key=len,reverse=True)
            special_pattern="|".join(re.escape(t) for t in sorted_special)
            self.special_regex=re.compile(special_pattern)
        else:
            self.special_regex=None
            
        # 4.预分词正则表达式
        self.gpt2_pat=re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        
        
    def encode(self,text:str)->list[int]:
        """
        将输入的原始字符串编码为整数 ID 列表。
        
        该方法的核心逻辑是：
        1. 作为一个“协调者”，它负责处理文本中的“特殊标记（Special Tokens）”和“普通文本”。
        2. 特殊标记（如 <|endoftext|>）被视为原子，直接映射为 ID，不参与 BPE 的拆分和合并。
        3. 普通文本片段则被交给底层逻辑执行预分词和 BPE 算法。
        
        参数:
            text: 需要编码的原始字符串（例如 "Hello<|end|>World"）。
            
        返回:
            list[int]: 编码后的整数 ID 序列。
        """ 
        
        # 1.边界情况检查
        if not text:
            return []
        
        # 2.情况a：快速路径
        if not self.special_regex:
            return self._encode_text_segment(text)
        
        # 3.情况b：处理含有特殊标记的复杂文本
        tokens=[]
        
        last_pos=0
        
        for match in self.special_regex.finditer(text):
            
            # 3.1提取并购处理”前置普通文本“
            pre_text=text[last_pos:match.start()]
            
            if pre_text:
                tokens.extend(self._encode_text_segment(pre_text))
                
            # 3.2处理当前特殊标记
            special_tok=match.group()
            
            tokens.append(self.byte_to_id[special_tok.encode('utf-8')])
            
            # 3.3更新游标
            last_pos=match.end()
            
            
        # 4.处理收尾文本
        remaining_text=text[last_pos:]
        if remaining_text:
            tokens.extend(self._encode_text_segment(remaining_text))
            
        return tokens
    
    def _encode_text_segment(self,text:str)->list[int]:
        """
        内部核心函数：对不含特殊 Token 的纯文本片段应用 BPE 合并逻辑。
        """
        
        ids=[]
        
        pre_tokens=self.gpt2_pat.findall(text)
        
        for p_tok in pre_tokens:
            byte_parts=[bytes([b]) for b in p_tok.encode("utf-8")]
            while len(byte_parts)>=2:
                best_pair=None
                min_rank=float('inf')
                
                # 寻找best pair
                for i in range(len(byte_parts)-1):
                    pair=(byte_parts[i],byte_parts[i+1])
                    if pair in self.merges:
                        rank =self.merges[pair]
                        if rank<min_rank:
                            min_rank=rank
                            best_pair=pair
                            
                #处理已经的确没有best pair的情况    
                if best_pair is None:
                    break
                
                # 开始合并
                new_byte_parts=[]
                i=0
                while i<len(byte_parts):
                    if i <len(byte_parts)-1 and (byte_parts[i],byte_parts[i+1])==best_pair:
                        new_byte_parts.append(best_pair[0]+best_pair[1])
                        i+=2
                    else:
                        new_byte_parts.append(byte_parts[i])
                        i+=1
                byte_parts=new_byte_parts
                
            # 将合并完成后的字节快转换为词表中的id
            for part in byte_parts:
                ids.append(self.byte_to_id[part])
        
        return ids
    
    def decode(self,ids:list[int])->str:
        """
        将 ID 列表解码为原始字符串。
        """
        # 1.根据Id查表找回字节快
        byte_segment=[self.id_to_byte[i] for i in ids]
        
        # 2.将所有字节快按顺序拼接成一个完整的字节流
        full_bytes=b''.join(byte_segment)
        
        # 3.将字节流解码为UTF-8字符串
        return full_bytes.decode('utf-8',errors='replace')
    
    def encode_iterable(self,iterable:Iterable[str])->Iterable[int]:
        """
        内存高效的迭代编码器。
        
        参数:
            iterable: 一个可迭代的字符串对象（例如文件句柄）。
        返回:
            一个生成器，逐个产出编码后的 ID。用于处理无法一次性读入内存的大文件。
        """
        for chunk in iterable:
            yield from self.encode(chunk)
        
                
        
        
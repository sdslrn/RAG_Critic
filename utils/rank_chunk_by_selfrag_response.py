

class Chunk_token:
    def __init__(self, index, IsREL, IsSUP):
        self.index = index
        self.IsREL = IsREL
        self.IsSUP = IsSUP

# response中有IsREL字段，表示是否为相关chunk，relevant为相关，irrelevant为不相关
# response中有IsSUP字段，表示是否为支持chunk，fully support为完全支持，partially support为部分支持，no support为不支持
# 从response中读取IsREL和IsSUP字段，然后根据IsREL字段对chunk进行排序，再根据IsSUP字段对chunk进行排序
# IsREL优先级高于IsSUP，即先根据IsREL排序，再根据IsSUP排序
def rank_chunk_by_selfrag_response(responses):
    """
    对检索到的chunk进行排序，返回chunk的下标（也为list）
    :param responses: 从selfrag返回的responses
    :return: chunk的下标
    """
    # 识别出response中的IsREL字段
    IsREL = [] # 1为relevant，0为irrelevant
    for response in responses:
        # 判断response字符串中是否有relevant或irrelevant，无论大小写
        if "relevant" in response.lower():
            IsREL.append(1)
        elif "irrelevant" in response.lower():
            IsREL.append(0)
        else:
            IsREL.append(-2)
        
    # 识别出response中的IsSUP字段
    IsSUP = [] # 1为fully support，0为partially support，-1为no support
    for response in responses:
        # 判断response字符串中是否有fully support或partially support或no support
        if "fully support" in response.lower():
            IsSUP.append(1)
        elif "partially support" in response.lower():
            IsSUP.append(0)
        elif "no support" in response.lower():
            IsSUP.append(-1)
        else:
            IsSUP.append(-2)
        # else:
        #     print("Error: response中没有fully support或partially support或no support")
            
    chunk_tokens = []
    for i in range(len(responses)):
        chunk_tokens.append(Chunk_token(i, IsREL[i], IsSUP[i]))
    # 对chunk_tokens进行排序，IsREL优先级高于IsSUP
    chunk_tokens.sort(key=lambda x: (x.IsREL, x.IsSUP), reverse=True)

    # 返回chunk的下标
    rank = []
    for chunk_token in chunk_tokens:
        rank.append(chunk_token.index)
    return rank

    

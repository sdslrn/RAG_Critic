
# Data Folder Structure

## Corpus
- row (原始语料)
- vector_database (向量数据库)
  - dragon_plus (构建方式)
    - medical (vdb名称)
    - education
  - msmarco
    - medical
    - education

## Datasets
- row (原始数据集)
  - medwiki (数据集)
  - medquad
- processed (预处理后的数据集)
  - medwiki.json
  - medquad.json
- retrieved (检索后的数据集)
  - medwiki
    - medical (检索的来源)
      - msmarco (检索的方法)
      - dragon_plus
      - self_rag
  - medquad
- recall(记录标准答案的排名)
  - medwiki
    - medical (检索的来源)
      - msmarco (检索的方法)
      - dragon_plus
  - medquad
    - medical
      - msmarco
      - dragon_plus

# dataset格式:
{
  id:int,
  query:str,
  answer:str,
}

{
  id:int,
  retrieved:[str]
}

{
  id:int,
  rank:int
}

# corpus格式:
{
  id:int,
  text:str
}
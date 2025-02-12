import torch
import sys
sys.path.append('/Users/stephensun/Desktop/pypro/LAMBDA/knowledge_integration')
from sentence_transformers import SentenceTransformer, util
import numpy as np
# from knw import KNW_INJECTION, knowledge_injection
from prompt_engineering.prompts import PMT_KNW_IN_CORE, PMT_KNW_IN_FULL
from knowledge_integration.nearest_correlation_matrix import nearest_correlation_matrix
from knowledge_integration.nn_network import nn_networks
from knowledge_integration.pami import pattern_mining
from kernel import execute


KNW_INJECTION = {}

def knowledge_register():
    ncm = nearest_correlation_matrix()
    ncm_key = ncm.name+ncm.description
    KNW_INJECTION[ncm_key] = ncm
    nnn = nn_networks()
    nnn_key = nnn.name+nnn.description
    KNW_INJECTION[nnn_key] = nnn
    pami = pattern_mining()
    pami_key = pami.name+pami.description
    KNW_INJECTION[pami_key] = pami



# 初始化句子嵌入模型
model = SentenceTransformer('all-MiniLM-L6-v2')

def search_knowledge(user_input, knowledge_embeddings, knowledge_keys):

    input_embedding = model.encode(user_input, convert_to_tensor=True) #embeding
    # similarity
    similarities_list = util.pytorch_cos_sim(input_embedding, knowledge_embeddings)
    if torch.max(similarities_list) > 0.3:

        best_match_idx = np.argmax(similarities_list.cpu())
        best_match_key = knowledge_keys[best_match_idx]
    else:
        best_match_key = False
    return (best_match_key, KNW_INJECTION[best_match_key]) if best_match_key else (False, None)


def format_code_snaps(knw, kernel):
    if knw.mode == 'full':
        core_code = knw.get_core_function()
        return PMT_KNW_IN_FULL.format(code=core_code)
    elif knw.mode == 'core':
        core_code = knw.get_core_function()
        runnable_code = knw.get_runnable_function()
        print("Knowledge_integration: core mode, runnable result: ", execute(runnable_code,kernel))
        retri_knw = PMT_KNW_IN_CORE.format(core=core_code, runnable=runnable_code)
        return retri_knw
    else:
        raise ValueError(f"Invalid mode: {knw.mode}, please choose from ['full', 'core'].")
        # test_case = knw.get_test_case()
        # return KNOWLEDGE_INJECTION_PMT_FIXED.format(test_case=test_case)


def retrieval_knowledge(instruction, kernel): # return code_snaps and mode: 'full' or runnable code in 'core'. Nothing retrieval, return None
    knowledge_register()
    knowledge_keys = list(KNW_INJECTION.keys())
    knowledge_embeddings = model.encode(knowledge_keys, convert_to_tensor=True)
    best_key, best_knw_object = search_knowledge(instruction, knowledge_embeddings, knowledge_keys)
    if best_key:
        return format_code_snaps(best_knw_object, kernel)
    else:
        return None

# def execute_runnable(code):
#     res_type, res = my_app.conv.run_code(code)


if __name__ == '__main__':
    # knowledge_register()
    # knowledge_keys = list(KNW_INJECTION.keys())
    # knowledge_embeddings = model.encode(knowledge_keys, convert_to_tensor=True)
    # user_input = "calculate nearest correlation matrix"
    # best_key, best_knw_object = search_knowledge(user_input)
    # print(best_key,best_knw_object)
    # print(f"Best match key: {best_key}")
    # print(format_code_snaps(best_knw_object))
    #print(retrieval_knowledge("calculate nearest correlation matrix", 'full'))
    print(retrieval_knowledge("Train a fixed points of nonnegative neural networks. Set parameters: networks: nn_sigmoid, learning rate: 5e-3, epochs: 30, wd: 0, b: 64"))
    #print(retrieval_knowledge("Use pattern mining to find frequent patterns in the dataset. Set parameters: fileURL: https://u-aizu.ac.jp/~udayrage/datasets/transactionalDatabases/Transactional_T10I4D100K.csv, minSup: 300."))
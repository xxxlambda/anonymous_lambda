import oss2
import time
import uuid
import os
from oss2.credentials import EnvironmentVariableCredentialsProvider
import requests
from joblib import load

bucket = None
expired_time = 3600


def init_dir(cache_dir):
    '''
    initialize the folders for conversation
    :return: the folder path of oss
    '''

    new_uuid = str(uuid.uuid4())
    current_fold = time.strftime('%Y-%m-%d', time.localtime())
    oss_file_dir = 'user_tmp/' + new_uuid + '-' + current_fold
    # local_cache_dir = '/home/maojsun/proj/dsagent_ci/cache/conv_cache/' + new_uuid + '-' + current_fold
    local_cache_dir = cache_dir + new_uuid + '-' + current_fold
    os.makedirs(local_cache_dir)
    return oss_file_dir, local_cache_dir


def init_oss(endpoint, access_key_id, access_key_secret, bucket_name, expired):
    auth = oss2.Auth(access_key_id, access_key_secret)
    global bucket
    bucket = oss2.Bucket(auth, endpoint, bucket_name)
    global expired_time
    expired_time = expired


def upload_oss_file(oss_file_dir, local_file_path):
    '''
    upload a local file to oss
    :param local_file_path: local file path
    :return: file name and download link, which can directly download the file by clicking the link.
    '''
    file_name = os.path.basename(local_file_path)

    object_name = os.path.join(oss_file_dir, file_name)

    bucket.put_object_from_file(object_name, local_file_path)
    download_link = get_download_link(object_name)
    return {'file_name': file_name, 'download_link': download_link}


def get_download_link(object_name):
    # auth = oss2.ProviderAuth(EnvironmentVariableCredentialsProvider())
    # 填写Object完整路径，例如exampledir/exampleobject.txt。Object完整路径中不能包含Bucket名称。 object_name = 'exampleobject.txt'
    # 生成下载文件的签名URL，有效时间为3600秒。 # 设置slash_safe为True，OSS不会对Object完整路径中的正斜线（/）进行转义，此时生成的签名URL可以直接使用。
    url = bucket.sign_url('GET', object_name, expired_time, slash_safe=True)
    return url


if __name__ == '__main__':
    # local_file = "/Users/stephensun/Desktop/pypro/darob/models/LogisticRegression.pkl"
    # model = load(local_file)
    # print(model)
    # upload_oss_file(local_file)
    before_files = os.listdir(
        '/Users/stephensun/Desktop/pypro/dsagent_ci/cache/conv_cache/6fa78267-4e0b-418e-ac47-c9d99b6bbe3b-2024-04-18')
    # with open('/Users/stephensun/Desktop/pypro/dsagent_ci/cache/conv_cache/6fa78267-4e0b-418e-ac47-c9d99b6bbe3b-2024-04-18/test.txt', 'w') as f:
    #     f.write("test")

    after_files = os.listdir(
        '/Users/stephensun/Desktop/pypro/dsagent_ci/cache/conv_cache/6fa78267-4e0b-418e-ac47-c9d99b6bbe3b-2024-04-18')
    # print(check_folder(before_files, after_files))

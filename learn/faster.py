from modelarts.session import Session

sess = Session()

if sess.region_name == 'cn-north-1':
    bucket_path = "modelarts-labs/notebook/DL_object_detection_faster/fasterrcnn.tar.gz"
elif sess.region_name == 'cn-north-4':
    bucket_path = "modelarts-labs-bj4/notebook/DL_object_detection_faster/fasterrcnn.tar.gz"
else:
    print("请更换地区到北京一或北京四")

sess.download_data(bucket_path=bucket_path, path="./fasterrcnn.tar.gz")
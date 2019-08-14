import os
import logging
import tempfile
from tqdm import tqdm
import requests
import shutil
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)


def fetch_pretrained_model(model_name, model_path):
    # Edited from https://github.com/huggingface/pytorch-pretrained-BERT/blob/68a889ee43916380f26a3c995e1638af41d75066/pytorch_pretrained_bert/file_utils.py
    # TODO: check whether the license from huggingface works with ours
    PRETRAINED_MODEL_ARCHIVE_MAP = {
        'politeness_model': ['http://jurgens.people.si.umich.edu/models/politeness_model.bin']
    }
    model = 'politeness_model'
    assert model_name in PRETRAINED_MODEL_ARCHIVE_MAP
    model_urls = PRETRAINED_MODEL_ARCHIVE_MAP[model_name]

    download_flag = False
    for idx, model_url in enumerate(model_urls):
        try:
            temp_file = tempfile.NamedTemporaryFile()
            logger.info(f'{model_path} not found in cache, downloading from {model_url} to {temp_file.name}')

            req = requests.get(model_url, stream=True)
            content_length = req.headers.get('Content-Length')
            total = int(content_length) if content_length is not None else None
            progress = tqdm(unit="KB", total=round(total / 1024))
            for chunk in req.iter_content(chunk_size=1024):
                if chunk:  # filter out keep-alive new chunks
                    progress.update(1)
                    temp_file.write(chunk)
            progress.close()
            temp_file.flush()
            temp_file.seek(0)
            download_flag = True
        except Exception as e:
            logger.warning(f'Download from {idx + 1}/{len(model_urls)} mirror failed with an exception of\n{str(e)}')
            try:
                temp_file.close()
            except Exception as e_file:
                logger.warning(f'temp_file failed with an exception of \n{str(e_file)}')
            continue

        if not download_flag:
            logging.warning(f'Download from all mirrors failed. Please retry.')
            return

        logger.info(f'Model {model_name} was downloaded to a tmp file.')
        logger.info(f'Copying tmp file to {model_path}.')
        with open(model_path, 'wb') as cache_file:
            shutil.copyfileobj(temp_file, cache_file)
        logger.info(f'Copied tmp model file to {model_path}.')
        temp_file.close()


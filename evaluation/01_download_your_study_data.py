import os
from urllib.request import urlretrieve
from datetime import datetime
from pathlib import Path

from zenlog import log


def download_user_study_data(url_, base_path='./data', file_name='user_study_feedback.json', datetime_str=''):

    file_path = Path(base_path).joinpath(file_name)
    if len(datetime_str) > 0:
        file_path = file_path.with_name(f'{file_path.stem}_from_{datetime_str}{file_path.suffix}')
    log.info(f'Downloading file from "firebaseio.com" to {file_path}')
    urlretrieve(url_, file_path)

    file_path_compressed = file_path.with_suffix('.7z')
    final_file_path = file_path.with_name(file_name)

    if file_path_compressed.is_file():
        file_path_compressed.unlink()

    # Note: "-mx=0" (store, fast) to "-mx=9" (small, slow) selects compression strength
    cmd = f'7z a -t7z -mx=9 {file_path_compressed} {file_path}'
    log.debug(f'Command line: {cmd}')
    log.info(f'Compress to file {file_path_compressed}')

    log.info(f'Save downloaded JSON file as {final_file_path} and {file_path_compressed}')

    r = os.system(cmd)
    if 'nt' != os.name and 0 != r:
        log.error(f'Compression of {file_path} to {file_path_compressed} failed. Skipping.')

    if final_file_path.is_file():
        final_file_path.unlink()

    os.rename(file_path, final_file_path)


if __name__ == '__main__':

    _url = r'https://project-....firebaseio.com/.json?print=pretty&format=export&' + \
           r'download=project-...'  # TODO Paste in your own firebase URL here
    current_day_str_for_file_name = datetime.now().strftime('%Y-%m-%d')

    log.info(f'Interaction input data URL: "{_url}"')
    try:
        download_user_study_data(_url, datetime_str=current_day_str_for_file_name)
    except UnicodeError:
        log.error('Paste in your own valid firebase URL in the script before running it')

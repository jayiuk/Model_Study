#!/usr/bin/python

import sys
import os
from configparser import ConfigParser
import ast

class Configuration(object):
    def __init__(self, file_name):
        parser = ConfigParser()
        parser.optionxform = str
        # 스크립트가 위치한 디렉터리의 절대 경로를 구합니다.
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # 스크립트 디렉터리에 있는 setup.ini 파일의 경로를 구성합니다.
        file_path = os.path.join(script_dir, file_name)
        found = parser.read(file_path)
        if not found:
            raise ValueError(f'No config file found at {file_path}!')
        for name in parser.sections():
            # ast.literal_eval()을 사용해 문자열 값을 평가하여 Python 객체로 변환합니다.
            self.__dict__.update({item[0]: ast.literal_eval(item[1]) for item in parser.items(name)})
        # base_dir을 현재 스크립트의 디렉터리로 설정합니다.
        self.base_dir = script_dir
config = Configuration('setup.ini')

#!/bin/bash
### 
# @Description: 用来同步本项目代码到服务器
 # @Author: Lai
 # @Date: 2019-08-22 13:07:14
 # @LastEditTime : 2020-01-15 16:11:17
 # @LastEditors  : Lai
###
# 获取当前目录的绝对路径
src=$(cd `dirname $0`; pwd)
target=$1
echo "source" $src
echo "target" $target/${src##*/}
# 和gitignore使用相同的排除文件，但是这里把vscode的配置文件也同步一下
exclude_file='./.gitignore'
# rsync -aztv --delete --include '.vscode/' --exclude-from $exclude_file  $src $target
rsync -aztv --delete --include '.vscode/' --exclude '.git'  $src $target
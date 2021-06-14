## Markdown的简单介绍
Markdown 是一种**轻量级标记语言**，它允许人们使用易读易写的纯文本格式编写文档。  
Markdown 编写的文档可以导出 HTML 、Word、图像、PDF、Epub 等多种格式的文档。  
Markdown 编写的文档后缀为 **.md**, **.markdown**。  
当前许多网站都广泛使用 Markdown 来撰写帮助文档或是用于论坛上发表消息。例如：**GitHub**、**简书**、reddit、Diaspora、Stack Exchange、OpenStreetMap 、SourceForge等。  
可使用 [**Typora**](https://typora.io/) 编辑器使用 Markdown，Typora 支持 MacOS 、Windows、Linux 平台，且包含多种主题，编辑后直接渲染出效果。    

## Markdown的语法规则
### 标题
使用 **#** 号可表示 1-6 级标题，一级标题对应一个 # 号，二级标题对应两个 # 号，以此类推。  
### 段落格式
段落的换行是使用两个以上空格加上回车。也可以在段落后面使用一个空行来表示。   
Markdown 可以使用以下几种字体:：`*斜体文本*`，`**粗体文本**`，`***粗斜体文本***`。 
可以在一行中用三个以上的星号、减号、底线来建立一个分隔线，行内不能有其他东西。  
在文字的两端加上两个波浪线 `~~` 和 HTML 的 `<u>` 可以分别添加删除线和下划线。  
脚注是对文本的补充说明，可使用`[^要注明的文本]`添加脚注。类似Datawhale [^Datawhale]。

[^Datawhale]: 一个热爱学习的社区
脚注说明在代码中可放在任意位置，例如`[^Datawhale]: 一个热爱学习的社区`，展示时会放到文档末尾。
### 列表
无序列表使用星号 `*` 、加号 `+` 或是减号 `-` 作为列表标记，添加一个空格后再填写内容。
有序列表使用数字并加上 `.` 号来表示。
列表嵌套只需在子列表中的选项前面添加四个空格即可。
### 区块
区块引用是在段落开头使用 `>` 符号 ，然后紧跟一个空格符号后再填写内容。
区块是可以嵌套的，一个 `>` 符号是最外层，两个 `>` 符号是第一层嵌套，以此类推。
区块中可以使用列表，在 `>` 符号后紧跟列表标记即可。
列表中可以使用区块，需要在 `>` 前添加四个空格的缩进。
### 代码
如果是段落上的一个函数或片段的代码可以用反引号把它包起来，例如：`printf()`函数。
代码区块使用 4 个空格或者一个制表符（Tab 键）。你也可以用三个反引号包裹一段代码，并指定一种语言（也可以不指定），显示效果如下：
```javascript
$(document).ready(function () {
    alert('RUNOOB');
});
```
### 链接
`[`链接名称`]`(链接地址) 或者 <链接地址>  效果如下所示：
这是一个链接 [菜鸟教程](https://www.runoob.com)
直接使用链接地址：<https://www.runoob.com>
### 图片
Markdown 图片语法格式如下：
`![alt 属性文本](图片地址)`
`![alt 属性文本](图片地址 "可选标题")`
使用实例：
`![RUNOOB 图标](http://static.runoob.com/images/runoob-logo.png "RUNOOB")`
![RUNOOB 图标](http://static.runoob.com/images/runoob-logo.png "RUNOOB")
也可使用本地路径 `![RUNOOB 图标](images/runoob-logo.png "RUNOOB")`
![RUNOOB 图标](images/runoob-logo.png "RUNOOB")
Markdown 还没有办法指定图片的高度与宽度，如果你需要的话，你可以使用普通的 `<img>` 标签。例如：`<img src="http://static.runoob.com/images/runoob-logo.png" width="30%">`。
<img src="http://static.runoob.com/images/runoob-logo.png" width="30%">

### 表格
Markdown 制作表格使用 `|` 来分隔不同的单元格，使用 `-` 来分隔表头和其他行。
语法格式如下：

>`|表头|表头|`
>`|----|----|`
>`|单元格|单元格|`
>`|单元格|单元格|`  

我们可以设置表格的对齐方式：
* `-:` 设置内容和标题栏居右对齐。
* `:-` 设置内容和标题栏居左对齐。
* `:-:` 设置内容和标题栏居中对齐。
实例如下：
>`|左对齐|右对齐|居中对齐|`
>`|:-----|----:|:----:|`
>`|单元格|单元格|单元格|`
>`|单元格|单元格|单元格|` 

|左对齐|右对齐|居中对齐|
|:-----|----:|:----:|
|单元格|单元格|单元格|
|单元格|单元格|单元格|
### 高级技巧
不在 Markdown 涵盖范围之内的标签，都可以直接在文档里面用 HTML 撰写。
目前支持的 HTML 元素有：`<kbd>` `<b>` `<i>` `<em>` `<sup>` `<sub>` `<br>`等。
Markdown 使用了很多特殊符号来表示特定的意义，如果需要显示特定的符号则需要使用转义字符，Markdown 使用反斜杠`\\`转义特殊字符。
当你需要在编辑器中插入数学公式时，可以使用两个美元符 $$ 包裹 TeX 或 LaTeX 格式的数学公式来实现。提交后，问答和文章页会根据需要加载 Mathjax 对数学公式进行渲染。
可以使用Typora绘制各种各样的流程图，展示效果如下所示：
```mermaid
graph LR
A[方形] -->B(圆角)
    B --> C{条件a}
    C -->|a=1| D[结果1]
    C -->|a=2| E[结果2]
    F[横向流程图]
```
## Markdown的应用场景
Markdown文档可以使用git进行版本管理，而word之类的文档不是很方便。
Markdown文档在编辑的时候可以较少考虑格式问题，鼠标使用频率低，可以专注写作提高效率。
Markdown文档在编辑公式的时候输出质量和效率都较高。
Markdown文档可以导出Word、PDF等多种格式，也方便与其他人进行交流。
## 参考资料
1. [Markdown教程](https://www.runoob.com/markdown/md-tutorial.html)

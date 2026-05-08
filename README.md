# agent-study
1. 黑盒
```plantuml
@startuml
start
:用户输入请求;
if (判断逻辑) then (通过)
:执行任务;
else (拒绝)
:返回错误;
endif
stop
@endum
```

2. 白盒

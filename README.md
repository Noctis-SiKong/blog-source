# Noctis 手记

在本机的 `D:\Obsidian-Hugo\myblog` 目录运行 `..\bin\hugo.exe server`，打开命令输出的本地地址预览。部署工作流使用 Hugo 0.152.2。

## 继续发文章

把 Markdown 放进 `content/post/`。也可以运行 `..\bin\hugo.exe new content post/文章标题.md` 生成草稿，发布时将 `draft` 设为 `false`。

```yaml
---
title: '文章标题'
date: '2025-12-25T19:30:10+08:00'
draft: false
categories: ['数据结构']
tags: ['二叉树', '遍历']
series: '数据结构与算法优化'
---
```

`date` 决定文章显示的日期、时间排序和归档月份。只需要日期时可写成 `date: '2025-12-25'`。未来发布日期遵循 Hugo 的发布规则，部署构建时已到该日期的文章才会进入站点。

`categories` 用于文章的大类，`tags` 用于知识点，`series` 用于连续更新的专栏。三个字段可选，名称相同的文章会自动归到同一个索引。文章开头可以写一段简介，再插入 `<!--more-->` 划定摘要范围。

## 前端入口

页面模板在 `layouts/`，配色、排版和动效在 `assets/blog.css`，回退、搜索和阅读进度在 `assets/blog.js`。现有 GitHub Actions 工作流在源码推送到 `main` 后构建并发布。

# Highlight.js

Highlight.js — это подсветчик синтаксиса, написанный на JavaScript. Он работает
и в браузере, и на сервере. Он работает с практически любой HTML разметкой, не
зависит от каких-либо фреймворков и умеет автоматически определять язык.


## Начало работы

Минимум, что нужно сделать для использования highlight.js на веб-странице — это
подключить библиотеку, CSS-стили и вызывать [`initHighlightingOnLoad`][1]:

```html
<link rel="stylesheet" href="/path/to/styles/default.css">
<script src="/path/to/highlight.pack.js"></script>
<script>hljs.initHighlightingOnLoad();</script>
```

Библиотека найдёт и раскрасит код внутри тегов `<pre><code>`, попытавшись
автоматически определить язык. Когда автоопределение не срабатывает, можно явно
указать язык в атрибуте class:

```html
<pre><code class="html">...</code></pre>
```

Список поддерживаемых классов языков доступен в [справочнике по классам][8].
Класс также можно предваоить префиксами `language-` или `lang-`.

Чтобы отключить подсветку для какого-то блока, используйте класс `nohighlight`:

```html
<pre><code class="nohighlight">...</code></pre>
```

## Инициализация вручную

Чтобы иметь чуть больше контроля за инициализацией подсветки, вы можете
использовать функции [`highlightBlock`][2] и [`configure`][3]. Таким образом
можно управлять тем, *что* подсвечивать и *когда*.

Вот пример инициализация, эквивалентной вызову [`initHighlightingOnLoad`][1], но
с использованием jQuery:

```javascript
$(document).ready(function() {
  $('pre code').each(function(i, block) {
    hljs.highlightBlock(block);
  });
});
```

Вы можете использовать любые теги разметки вместо `<pre><code>`. Если
используете контейнер, не сохраняющий переводы строк, вам нужно сказать
highlight.js использовать для них тег `<br>`:

```javascript
hljs.configure({useBR: true});

$('div.code').each(function(i, block) {
  hljs.highlightBlock(block);
});
```

Другие опции можно найти в документации функции [`configure`][3].


## Установка библиотеки

Highlight.js можно использовать в браузере прямо с CDN хостинга или скачать
индивидуальную сборку, а также установив модуль на сервере. На
[страница загрузки][4] подробно описаны все варианты.

Обратите внимание, что библиотека не предназначена для использования в виде
исходного кода на GitHub, а требует отдельной сборки. Если вам не подходит ни
один из готовых вариантов, читайте [документацию по сборке][5].


## Лицензия

Highlight.js распространяется под лицензией BSD. Подробнее читайте файл
[LICENSE][10].


## Ссылки

Официальный сайт билиотеки расположен по адресу <https://highlightjs.org/>.

Более подробная документация по API и другим темам расположена на
<http://highlightjs.readthedocs.org/>.

Авторы и контрибьютора перечислена в файле [AUTHORS.ru.txt][9] file.

[1]: http://highlightjs.readthedocs.org/en/latest/api.html#inithighlightingonload
[2]: http://highlightjs.readthedocs.org/en/latest/api.html#highlightblock-block
[3]: http://highlightjs.readthedocs.org/en/latest/api.html#configure-options
[4]: https://highlightjs.org/download/
[5]: http://highlightjs.readthedocs.org/en/latest/building-testing.html
[8]: http://highlightjs.readthedocs.org/en/latest/css-classes-reference.html
[9]: https://github.com/isagalaev/highlight.js/blob/master/AUTHORS.ru.txt
[10]: https://github.com/isagalaev/highlight.js/blob/master/LICENSE

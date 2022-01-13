window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  },
  startup: {
    typeset: true, // because we load MathJax asynchronously
  },
  options: {
    processHtmlClass: "arithmatex"
  },
  chtml: {
    mtextInheritFont: true,       // true to make mtext elements use surrounding font
  }
};

document$.subscribe(() => {
  MathJax.typesetPromise()
})

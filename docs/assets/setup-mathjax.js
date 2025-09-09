window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"], ["$", "$"]],
    displayMath: [["\\[", "\\]"], ["$$", "$$"]],
    processEscapes: true,
    processEnvironments: true
  },
  startup: {
    ready: () => {
      MathJax.startup.defaultReady();
      MathJax.startup.promise.then(() => {
        document$.subscribe(() => {
          MathJax.typesetPromise()
        })
      });
    }
  }
};
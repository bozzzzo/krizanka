1. **Settings** Used with preprocessors and contain font, colors definitions, etc. In this layer is common define the variables that can customize the template.

2. **Tools** — globally used mixins and functions. This layer is only used if we use a preprocessor as SASS.
3. **Generic** — reset and/or normalize styles, box-sizing definition, etc. It is important to note that this is the first layer of the triangle that generates CSS.
4. **Elements** — styling for bare HTML elements (like H1, A, header, footer, …). These come with default styling from the browser so we must to redefine them here.
5. **Objects** — class-based selectors which define undecorated design patterns, for example media object known from OOCSS such as list, radio-button. The Daniel Fornell’s codepen <a href="https://codepen.io/collection/DmzVOM/">https://codepen.io/collection/DmzVOM/</a> shows you a list of objects designed using this architecture.
6. **Components** — specific UI components. The components of our page, for example button, card, concrete-list, etc..
7. **Utilities** — utilities and helper classes with ability to override anything which goes before in the triangle.

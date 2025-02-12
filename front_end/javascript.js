
function createGradioAnimation() {
    var container = document.createElement('div');
    container.id = 'gradio-animation';
    container.style.fontSize = '2em';
    container.style.fontWeight = 'bold';
    container.style.textAlign = 'center';
    container.style.marginBottom = '20px';

    var text = 'Welcome to LAMBDA! Easy Data Analytics!';
    for (var i = 0; i < text.length; i++) {
        (function(i){
            setTimeout(function(){
                var letter = document.createElement('span');
                letter.style.opacity = '0';
                letter.style.transition = 'opacity 0.5s';
                letter.innerText = text[i];

                container.appendChild(letter);

                setTimeout(function() {
                    letter.style.opacity = '1';
                }, 50);
            }, i * 250);
        })(i);
    }

    var gradioContainer = document.querySelector('.gradio-container');
    gradioContainer.insertBefore(container, gradioContainer.firstChild);


    var ed_btn = document.querySelector('.ed_btn');
    var ed  = document.querySelector('.ed');
    //console.log(button)
    ed_btn.addEventListener('click', function() {
        ed.style.display = 'block';
    });

    var button = document.querySelector('.df_btn');
    var df = document.querySelector('.df');
    //console.log(button)
    button.addEventListener('click', function() {
        df.style.display = 'block';
    });



    return 'Animation created';
}

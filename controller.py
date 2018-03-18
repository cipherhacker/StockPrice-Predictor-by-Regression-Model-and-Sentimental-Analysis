from model import InputForm
from flask import Flask, render_template, request
from compute import predict
import sys

app = Flask(__name__)

try:
    template_name = sys.argv[1]
except IndexError:
    template_name = 'view'

if template_name == 'view_flask_bootstrap':
    from flask_bootstrap import Bootstrap
    Bootstrap(app)

@app.route('/Flask_project', methods=['GET', 'POST'])
def index():
	
    form = InputForm(request.form)
    if request.method == 'POST' and form.validate():
        result1,result2 = predict(form.x.data,form.open.data, form.high.data,
                         form.low.data, form.last.data,
						 )
    else:
        result = None
    print (form, dir(form))
    #print form.keys()
    for f in form:
        print (f.id)
        print (f.name)
        print (f.label)

    return render_template(template_name + '.html',
                           form=form, result1=result1,result2=result2)

if __name__ == '__main__':
    app.run(debug=True)
from flask import Flask,redirect, url_for, render_template

# WSGI Application
app = Flask(__name__)

@app.route('/')
def welcome():
    return render_template('index.html')

@app.route('/success/<int:score>')
def success(score):
    return "The person passed with  " + str(score)

@app.route('/fail/<int:score>')
def fail(score):
    return "The person fail with  " + str(score)

@app.route('/results/<int:marks>')
def results(marks):
    result =""
    if marks <50:
        result = "fail"
    else:
        result = "success"    
    return redirect(url_for(result,score=marks))

if __name__=='__main__':
    app.run(debug=True)
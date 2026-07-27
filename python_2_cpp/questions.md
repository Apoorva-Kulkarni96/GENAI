"What is the difference between GET and POST?"

"GET is used to retrieve data without changing anything on the server — it's idempotent. POST is used to send data to the server to create or trigger something. In our project, I use GET for fetching system info and POST for sending Python code to convert because we're sending a body of data and triggering a complex operation."

"What does a 400 status code mean?"

"400 means Bad Request — the client sent data the server couldn't understand or that failed validation. In our project, if someone sends an empty python_code field, we return a 400 with a message explaining what went wrong, rather than passing the empty string to Ollama."

"What is JSON?"

"JSON stands for JavaScript Object Notation — it's a text format for representing structured data as key-value pairs. It's language-agnostic, so a Python server and a JavaScript browser can both read and write it. Under the hood, it's just a string that follows specific formatting rules."


Why /api/system-info and not just /system-info?
Convention. The /api/ prefix tells anyone reading your code: "these are data endpoints, not pages." Routes without /api/ serve HTML pages. Routes with /api/ serve JSON data. You'll see this pattern in every professional project.

"What is a decorator in Python, and how does FastAPI use it?"

"A decorator is a function that wraps another function to add behaviour. FastAPI uses decorators like @app.get('/') to register a function as a route handler — it tells the framework: when a GET request arrives at this path, call this function and return its result as an HTTP response."

"What is Pydantic and why do you use it?"

"Pydantic is a data validation library. I use it to define the expected shape of incoming request data. If the request is missing a required field or has the wrong type, Pydantic automatically returns a 422 error with a detailed message. This means I never write manual validation code — the class definition is the contract."

"What's the difference between @app.get and @app.post?"

"GET is for retrieving data — it has no request body. POST is for sending data to the server — it carries a JSON body. I use GET for /api/system-info because I'm just reading hardware info, and POST for /api/convert because I'm sending Python code that the server needs to process."

<html>

<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Classifier</title>
    <link
      rel="icon"
      type="image/png"
      href="https://www.google.com/s2/favicons?domain=gnanaganesh-resume.web.app"
    />
    <link
      rel="stylesheet"
      href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.12.0-2/css/all.min.css"
    />
    <link rel="stylesheet" type="text/css" href="{{url_for('static',filename='styles.css')}}" />
</head>


          
   
<body>
         


   <form id="imgur" method="POST" enctype="multipart/form-data" >


   <div class="quote-container" id="quote-container">
      <div class="quote-text">
        <span id="quote">

          


         {% if data %}
         <h4 id="text">Your Crop name:</h4><p>{{ data }}</p>
          {% endif %}

        </span>
      </div>
  </div>

     
     
</form>

</body>
    <script src="scripts.js"></script>





<script src="https://cdn.jsdelivr.net/npm/jquery@3.2.1/dist/jquery.min.js"></script>
<script>
    $("document").ready(function () {

        $('input[type=file]').on("change", function () {

            var $files = $(this).get(0).files;

            if ($files.length) {

                // Reject big files
                if ($files[0].size > $(this).data("max-size") * 1024) {
                    console.log("Please select a smaller file");
                    return false;
                }

                // Replace ctrlq with your own API key
                var apiUrl = 'https://api.imgur.com/3/image';
                var apiKey = '5b103bbe3e1ff42';

                var formData = new FormData();
                formData.append("image", $files[0]);

                var settings = {
                    "async": true,
                    "crossDomain": true,
                    "url": apiUrl,
                    "method": "POST",
                    "datatype": "json",                
                    "headers": {
                        "Authorization": "Client-ID " + apiKey
                    },
                    "processData": false,
                    "contentType": false,
                    "data": formData,
                    beforeSend: function (xhr) {
                        console.log("Uploading | 上传中");
                    },
                    success: function (res) {
                        $('form').append('<input type="hidden" name="hidden" value="'+res.data.link+'">');
                                                 console.log(res.data.link);
      },
                    error: function () {
                        alert("Failed | 上传失败");
                    }
                }
                $.ajax(settings).done(function (response) {
                    console.log("Done | 成功");
                });
            }
        });
    });
</script>


        
    </div>

    

</div>
</html>

import { Component, OnInit } from '@angular/core';
import { UploadEvent, UploadFile, FileSystemFileEntry, FileSystemDirectoryEntry } from 'ngx-file-drop';
import { ModelFile } from '../modelfile';
import { ModelService } from '../model.service';
import { CLASSIFIERS } from '../classifiers';
import { Classifier } from '../classifier';
import { Router, NavigationExtras } from '@angular/router';

@Component({
  selector: 'app-data-add',
  templateUrl: './data-add.component.html',
  styleUrls: ['./data-add.component.less']
})
export class DataAddComponent implements OnInit {
  classifiers = CLASSIFIERS;
  selectedClassifier: Classifier;
  fileToUpload : File;
  name: string = "";
  error: string;

  constructor(public modelService: ModelService, public router : Router) {}

  ngOnInit() {
  }

  private updateName() : void {
    if (this.selectedClassifier && this.fileToUpload && this.name == "")
    {
      var fileName = this.fileToUpload.name;
      var extensionIndex = fileName.lastIndexOf(".")
      console.log("File looks like " + fileName);

      if (extensionIndex > 0)
      {
        fileName = fileName.substring(0, extensionIndex)
        console.log("File looks like " + fileName);
      }

      this.name = this.selectedClassifier.name + "_" + fileName
    }
  }

  public onSelect(classifier: Classifier) : void {
    this.selectedClassifier = classifier;
    this.updateName();
  }

  public fileDropped(event: UploadEvent) {
    for (const droppedFile of event.files) {

      // Is it a file?
      if (droppedFile.fileEntry.isFile) {
        const fileEntry = droppedFile.fileEntry as FileSystemFileEntry;
        fileEntry.file((file: File) => {
 
          // Here you can access the real file
          console.log(droppedFile.relativePath, file);

          this.fileToUpload = file;
          this.updateName();
          
          /**
          // You could upload it like this:
          const formData = new FormData()
          formData.append('logo', file, relativePath)
 
          // Headers
          const headers = new HttpHeaders({
            'security-token': 'mytoken'
          })
 
          this.http.post('https://mybackend.com/api/upload/sanitize-and-save-logo', formData, { headers: headers, responseType: 'blob' })
          .subscribe(data => {
            // Sanitized logo returned from backend
          })
          **/
 
        });
      } else {
        // It was a directory (empty directories are added, otherwise only files)
        const fileEntry = droppedFile.fileEntry as FileSystemDirectoryEntry;
        console.log(droppedFile.relativePath, fileEntry);
      }
    }
  }

  public onSubmit() : void {
    this.error = null;
    this.modelService.trainClassifier(this.selectedClassifier, this.fileToUpload, this.name).subscribe(data => {
      let navigationExtras : NavigationExtras = {
        queryParams: {
          result: JSON.stringify(data)
      }};

      this.router.navigate(["/train-result"], navigationExtras);
    },
    err => {
      this.error = "Unable to complete your request. Please try again."
    });
  }
}

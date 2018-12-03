import { Component, OnInit, ViewChild } from '@angular/core';
import { Model } from '../model';
import { Classification } from '../classification';
import { ModelService } from '../model.service';
import { UploadEvent, FileSystemFileEntry } from 'ngx-file-drop';
import { GridOptions, RowNode } from 'ag-grid-community';
import { Classifier } from '../classifier';
import { HttpClient } from '@angular/common/http';
import { AgGridNg2 } from 'ag-grid-angular';
import { Observable, of } from 'rxjs';
import { map, tap } from 'rxjs/operators';
import { ClassifierResult } from '../classifier-result';

class Result {
  value : string;
}


@Component({
  selector: 'app-test',
  templateUrl: './test.component.html',
  styleUrls: ['./test.component.less']
})
export class TestComponent implements OnInit {
  @ViewChild('agGrid') agGrid: AgGridNg2;

  models: Model[];
  selectedModel: Model;
  classifierResults: ClassifierResult;
  predictions : Classification[] = [];
  error: string;
  info: string;
  gridOptions: GridOptions;

  rowData:any =  [];
  columnDefs = [
    {headerName: 'Text', field: 'text' },
    {headerName: 'Prediction', field: 'prediction' },
    {headerName: 'Label', field: 'label'},
    {headerName: 'Result', field: 'result'}
  ];
  resultToFilter: Result =  {value: "All"};
  results :Result[] = [
    {
      value: "All"
    },
    {
      value: "Positive"
    },
    {
      value: "Negative"
    }
  ];

  constructor(private modelService : ModelService, private http: HttpClient) { 
  }

  ngOnInit() {
    this.gridOptions = {
      onGridReady: function (params) {
        params.api.sizeColumnsToFit();

        window.addEventListener('resize', function() {
          setTimeout(function() {
            params.api.sizeColumnsToFit();
          })
        })
      },
      animateRows: true,
      isExternalFilterPresent: this.isExternalFilterPresent.bind(this),
      doesExternalFilterPass: this.doesExternalFilterPass.bind(this)
    };

    this.getModels();
  }
  
  isExternalFilterPresent() : boolean {
    return this.resultToFilter.value != "All";
  }

  doesExternalFilterPass(node : RowNode) : boolean {
    return this.resultToFilter.value == "All"  ? true : node.data.result == this.resultToFilter.value;
  }

  updateFilter(filterValue : string) : void {
    this.gridOptions.api.onFilterChanged();
  }

  getModels(): void {
    this.modelService.getModels()
      .subscribe(models => { 
        this.models = models
        if (this.models.length > 0)
        {
          this.selectedModel = this.models[0]
        }
      });
  }

  private runTest(file : File) : void {
    console.log("Running test ");
    this.modelService.predict(file, this.selectedModel).subscribe(
      result => {
        this.info = null;
        this.predictions = result.classifications;
        this.gridOptions.api.setRowData(this.predictions);
        this.gridOptions.api.sizeColumnsToFit();
        this.classifierResults = result;
        console.log(this.classifierResults);
      }
    )
  }
  
  public dataFileDropped(event: UploadEvent) {
    for (const droppedFile of event.files) {
      if (droppedFile.fileEntry.isFile) {
        const fileEntry = droppedFile.fileEntry as FileSystemFileEntry;
        fileEntry.file((file: File) => {
          this.info = "Running model against test data "  + file.name + "..."
          this.runTest(file)
        });
      }
    }
  }
}
